"""What gets views: within-channel comparisons -> web/data/analysis.json (read by the UI's Insights report).

Every video is scored against its own channel's other videos from the same year and format (long / Short):
  lift = log2(views / median views of that channel-year-format), only where that group has >= 8 videos.
That removes channel size, channel growth and most of the "older videos had longer to collect views" effect.

  feature tests   for each channel with >= 5 videos with the feature and >= 5 without, the median lift of
                  each side; reported as the median gap across channels (as a multiplier) and the share of
                  channels where the feature side won. Title tricks, title length, video length, and
                  thumbnail looks (Gemini text phrases scored against the thumbnail vectors).
  topics by year  the clusters whose videos most beat their channel's usual, per year
  early vs late   in clusters that took off, a channel's videos by how early it got there vs the cluster's peak
  shorts vs long  for channels posting both in a year, their Shorts' median views / their long videos'
  engagement      likes per 1,000 views by year, long vs Short
  nice uv run python pipeline/analyze.py
"""
import glob, json, os, re
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, W = os.path.join(ROOT, "data"), os.path.join(ROOT, "web", "data")
PT = [("x", "<f4"), ("y", "<f4"), ("ch", "<u2"), ("m", "<u2"), ("cl", "<i2"), ("kind", "u1"), ("game", "u1")]

meta = json.load(open(os.path.join(W, "meta.json")))
r = np.fromfile(os.path.join(W, "points.bin"), dtype=PT)
N = len(r)
views = np.fromfile(os.path.join(W, "views.bin"), dtype="<u4").astype(np.float64)
titles = open(os.path.join(W, "titles.txt"), newline="").read().split("\n")[:N]
ids = open(os.path.join(W, "ids.txt")).read().split("\n")[:N]
det = {}
for p in glob.glob(os.path.join(D, "details", "*.jsonl")):
    for d in map(json.loads, open(p)):
        det[d["id"]] = d
dur = np.array([det.get(i, {}).get("dur", 0) or 0 for i in ids], dtype=np.float64)
likes = np.array([det.get(i, {}).get("likes", 0) or 0 for i in ids], dtype=np.float64)
year = r["m"] // 12 + 2005
mc = r["game"] == 0
names = {c["id"]: c["name"] for c in meta["clusters"]}
types = {c["id"]: c["type"] for c in meta["clusters"]}
chname = [c["name"] for c in meta["channels"]]

# ---- lift vs the channel-year-format median
lift = np.full(N, np.nan)
grp = defaultdict(list)
for i in np.flatnonzero(views > 0):
    grp[(r["ch"][i], year[i], r["kind"][i])].append(i)
for g in grp.values():
    if len(g) >= 8:
        g = np.array(g)
        lift[g] = np.log2(views[g]) - np.log2(np.median(views[g]))
ok = ~np.isnan(lift) & mc
print(f"{ok.sum():,} Minecraft videos with a lift score")
out = {"n": int(ok.sum())}


BYCH = np.split(np.argsort(r["ch"], kind="stable"), np.cumsum(np.bincount(r["ch"], minlength=len(chname)))[:-1])


RNG = np.random.default_rng(0)


def test(mask, base=None, sel=None, min_each=5, topic=False):
    """median over channels of (median lift with feature - median lift without), with a 95% interval from
    resampling channels. topic=True compares only within the same channel AND the same cluster (each channel's
    gap = the median of its per-cluster gaps), so a trick that just rides a popular topic ("100 Days") doesn't count."""
    sel = ok if sel is None else ok & sel
    base = ~mask if base is None else base
    gaps = []
    for c in BYCH:
        c = c[sel[c]]
        if len(c) < 2 * min_each:
            continue
        if not topic:
            a, b = lift[c[mask[c]]], lift[c[base[c]]]
            if len(a) >= min_each and len(b) >= min_each:
                gaps.append(np.median(a) - np.median(b))
            continue
        c = c[r["cl"][c] >= 0]
        g = []
        for k in np.unique(r["cl"][c]):
            ck = c[r["cl"][c] == k]
            a, b = lift[ck[mask[ck]]], lift[ck[base[ck]]]
            if len(a) >= 3 and len(b) >= 3:
                g.append((np.median(a) - np.median(b), min(len(a), len(b))))
        if sum(n for _, n in g) >= min_each:
            gaps.append(np.average([x for x, _ in g], weights=[n for _, n in g]))
    if len(gaps) < 15:
        return None
    gaps = np.array(gaps)
    boot = np.median(gaps[RNG.integers(0, len(gaps), (1000, len(gaps)))], axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"x": round(float(2 ** np.median(gaps)), 3), "lo": round(float(2 ** lo), 3), "hi": round(float(2 ** hi), 3),
            "win": round(float((gaps > 0).mean()), 3), "channels": len(gaps), "videos": int((sel & mask).sum())}


# ---- title tricks
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
words = [t.split() for t in titles]
capsr = np.array([sum(w.isupper() and len(w) > 1 for w in ws) / max(1, sum(len(w) > 1 for w in ws)) for ws in words])
nw = np.array([len(ws) for ws in words])
low = [t.lower() for t in titles]
F = {
    "Mostly CAPITALS": capsr >= 0.5,
    "A number in it": np.array([bool(re.search(r"\d", t)) for t in titles]),
    "100 or 1,000": np.array([bool(re.search(r"\b(100|1000|1,000)\b", t)) for t in titles]),
    "A question mark": np.array(["?" in t for t in titles]),
    "An exclamation mark": np.array(["!" in t for t in titles]),
    "An emoji": np.array([bool(EMOJI.search(t)) for t in titles]),
    "“vs”": np.array([bool(re.search(r"\bvs\.?\b", t)) for t in low]),
    "Starts with I or I'm": np.array([bool(re.match(r"(i|i'm|i've)\b", t.strip())) for t in low]),
    "Says “Minecraft”": np.array(["minecraft" in t for t in low]),
    "Episode or part number": np.array([bool(re.search(r"(\bep\.?\s*\d|episode\s*\d|#\d|\bpart\s*\d)", t)) for t in low]),
    "Brackets, like (gone wrong)": np.array([bool(re.search(r"[\(\[]", t)) for t in titles]),
    "“but”": np.array([bool(re.search(r"\bbut\b", t)) for t in low]),
}
ERAS = [("2010–15", 2010, 2015), ("2016–20", 2016, 2020), ("2021–26", 2021, 2026)]
tricks = []
for name, m in F.items():
    row = {"name": name, "all": test(m, sel=r["kind"] == 0), "topic": test(m, sel=r["kind"] == 0, topic=True)}
    row["eras"] = [test(m, sel=(r["kind"] == 0) & (year >= a) & (year <= b)) for _, a, b in ERAS]
    row["shorts"] = test(m, sel=r["kind"] == 1)
    if row["all"]:
        tricks.append(row)
tricks.sort(key=lambda t: -t["all"]["x"])
out["eras"] = [e[0] for e in ERAS]
out["title_tricks"] = tricks

# title length: short (<= 5 words) and long (>= 11) against the middle
out["title_length"] = {"short": test(nw <= 5, base=(nw >= 6) & (nw <= 10), sel=r["kind"] == 0),
                       "long": test(nw >= 11, base=(nw >= 6) & (nw <= 10), sel=r["kind"] == 0)}

# ---- video length (long videos): each bucket against the channel's 10-20 min videos
B = [("Under 5 min", 60, 300), ("5–10 min", 300, 600), ("20–40 min", 1200, 2400), ("40 min–1 h", 2400, 3600), ("Over 1 h", 3600, 1e9)]
mid = (dur >= 600) & (dur < 1200)
out["video_length"] = [{"name": n, **{"eras": [test((dur >= a) & (dur < b), base=mid, sel=(r["kind"] == 0) & (year >= e0) & (year <= e1))
                                               for _, e0, e1 in ERAS]}, "all": test((dur >= a) & (dur < b), base=mid, sel=r["kind"] == 0),
                        "topic": test((dur >= a) & (dur < b), base=mid, sel=r["kind"] == 0, topic=True)}
                       for n, a, b in B]

# ---- thumbnail looks: linear probes (look_probe.py) trained on 16k vision-model labels, checked on held-out labels
from look_probe import thumb_vecs
PR = np.load(os.path.join(D, "look_probes.npz"))
LOOK_NAMES = {"real_person": "A real person's face", "big_text": "Big text", "no_text": "No text at all", "red_marks": "Red arrows or circles",
              "dark": "Dark or scary", "two_sides": "A versus or split layout", "giant_thing": "One giant thing", "treasure": "Diamonds, gold or money",
              "character_focus": "One big character", "raw_gameplay": "A plain gameplay screenshot", "cartoon": "Drawn or animated art",
              "shocked_face": "A shocked face"}
tvec = thumb_vecs(ids)
has = np.array([i in tvec for i in ids])
TV = np.zeros((N, 768), np.float32)
TV[has] = np.stack([tvec[i] for i, h in zip(ids, has) if h])
del tvec
looks = []
for k, name in LOOK_NAMES.items():
    if k not in PR:
        continue
    w, (auc, acc, prec, rec, share, _) = PR[k], PR[k + "__stats"]
    if auc < 0.9:
        continue
    z = TV @ w[:768] + w[768]
    pr = 1 / (1 + np.exp(-z))
    m = has & (pr >= w[769])
    row = {"key": k, "name": name, "auc": round(float(auc), 3), "precision": round(float(prec), 2), "recall": round(float(rec), 2),
           "all": test(m, base=has & ~m), "long": test(m, base=has & ~m, sel=r["kind"] == 0), "shorts": test(m, base=has & ~m, sel=r["kind"] == 1),
           "topic": test(m, base=has & ~m, sel=r["kind"] == 0, topic=True),
           "eras": [test(m, base=has & ~m, sel=(r["kind"] == 0) & (year >= a) & (year <= b)) for _, a, b in ERAS]}
    # how common, per year: mean probability (an unbiased share estimate when the probe is calibrated), Minecraft only
    row["trend"] = [round(float(pr[mc & has & (year == y)].mean()), 4) for y in range(2011, 2027)]
    looks.append(row)
    print(f"  look {name}: {row['long']}  same topic {row['topic']}")
out["looks_labelled"] = sum(1 for _ in open(os.path.join(D, "thumb_labels.jsonl")))
out["looks"] = sorted([l for l in looks if l["long"]], key=lambda l: -l["long"]["x"])
out["years"] = list(range(2011, 2027))

# ---- topics that beat their channel's usual, per year
by_year = []
for y in range(2011, 2027):
    rows = []
    for k in names:
        if types.get(k) != "mc":
            continue
        s = ok & (year == y) & (r["cl"] == k)
        if s.sum() < 60 or len(np.unique(r["ch"][s])) < 5:
            continue
        rows.append((float(np.median(lift[s])), k, int(s.sum()), int(len(np.unique(r["ch"][s])))))
    rows.sort(reverse=True)
    by_year.append({"year": y, "top": [{"id": k, "name": names[k], "x": round(2 ** v, 2), "n": n, "channels": c} for v, k, n, c in rows[:3]],
                    "bottom": [{"id": k, "name": names[k], "x": round(2 ** v, 2), "n": n, "channels": c} for v, k, n, c in rows[-2:]]})
out["topics_by_year"] = by_year

# ---- early vs late into a trend
ins = json.load(open(os.path.join(W, "insights.json")))
rel = defaultdict(list)
for s in ins["starters"]:
    k, peak = s["id"], s["peak"]
    idx = np.flatnonzero(ok & (r["cl"] == k))
    first = {}
    for i in idx[np.argsort(r["m"][idx])]:
        first.setdefault(int(r["ch"][i]), int(year[i]))
    for i in idx:
        d = first[int(r["ch"][i])] - peak  # when this channel arrived, in years from the peak
        b = "2+ years early" if d <= -2 else "1 year early" if d == -1 else "In the peak year" if d == 0 else "After the peak"
        rel[b].append((int(r["ch"][i]), lift[i]))
order = ["2+ years early", "1 year early", "In the peak year", "After the peak"]


def by_channel(pairs):
    """median over channels of each channel's median lift, with a 95% interval from resampling channels"""
    per = defaultdict(list)
    for c, v in pairs:
        per[c].append(v)
    g = np.array([np.median(v) for v in per.values() if len(v) >= 3])
    boot = np.median(g[RNG.integers(0, len(g), (1000, len(g)))], axis=1)
    return {"x": round(float(2 ** np.median(g)), 3), "lo": round(float(2 ** np.quantile(boot, .025)), 3),
            "hi": round(float(2 ** np.quantile(boot, .975)), 3), "channels": len(g), "videos": len(pairs)}


out["early_late"] = [{"name": b, **by_channel(rel[b])} for b in order if len(rel[b]) > 30]

# ---- Shorts vs long, same channel same year
sl = []
for y in range(2020, 2027):
    ratios = []
    for c in BYCH:
        c = c[(mc & (year == y) & (views > 0))[c]]
        a, b = views[c[r["kind"][c] == 1]], views[c[r["kind"][c] == 0]]
        if len(a) >= 5 and len(b) >= 5:
            ratios.append(np.log2(np.median(a) / np.median(b)))
    if len(ratios) >= 10:
        sl.append({"year": y, "x": round(float(2 ** np.median(ratios)), 2), "win": round(float((np.array(ratios) > 0).mean()), 3), "channels": len(ratios)})
out["shorts_vs_long"] = sl

# ---- likes per 1,000 views
eng = {"years": list(range(2011, 2027)), "long": [], "short": []}
for y in eng["years"]:
    for kd, key in ((0, "long"), (1, "short")):
        s = mc & (year == y) & (r["kind"] == kd) & (views > 1000) & (likes > 0)
        eng[key].append(round(float(np.median(likes[s] / views[s] * 1000)), 1) if s.sum() > 200 else None)
out["engagement"] = eng

json.dump(out, open(os.path.join(W, "analysis.json"), "w"), ensure_ascii=False)
print("\nTITLE TRICKS (long videos, x vs same channel's other videos, share of channels where it wins)")
for t in tricks:
    print(f"  {t['name']:28s} {t['all']['x']:.2f}x [{t['all']['lo']:.2f}-{t['all']['hi']:.2f}]  same topic {t['topic'] and (t['topic']['x'], t['topic']['lo'], t['topic']['hi'])}  wins {t['all']['win']:.0%}  ({t['all']['channels']} ch)  eras {[e and e['x'] for e in t['eras']]}  shorts {t['shorts'] and t['shorts']['x']}")
print("TITLE LENGTH", out["title_length"])
print("VIDEO LENGTH vs 10-20 min:")
for v in out["video_length"]:
    print(f"  {v['name']:12s} {v['all'] and (v['all']['x'], v['all']['lo'], v['all']['hi'])} same topic {v['topic'] and (v['topic']['x'], v['topic']['lo'], v['topic']['hi'])}  eras {[e and e['x'] for e in v['eras']]}")
print("LOOKS:")
for l in out["looks"]:
    L_ = l["long"]
    print(f"  {l['name']:30s} long {L_['x']:.2f}x [{L_['lo']:.2f}-{L_['hi']:.2f}] wins {L_['win']:.0%}  same topic {l['topic'] and (l['topic']['x'], l['topic']['lo'], l['topic']['hi'])}  shorts {l['shorts'] and (l['shorts']['x'], l['shorts']['lo'], l['shorts']['hi'])}  eras {[e and e['x'] for e in l['eras']]}  (AUC {l['auc']}, precision {l['precision']})")
print("EARLY/LATE", out["early_late"])
print("SHORTS vs LONG", sl)
print("ENGAGEMENT", eng)
for b in by_year:
    print(b["year"], [(t["name"], t["x"]) for t in b["top"]], "| worst", [(t["name"], t["x"]) for t in b["bottom"]])
