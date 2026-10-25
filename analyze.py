"""How Minecraft YouTube changed, described (not "what gets views") -> web/data/analysis.json, read by the Insights report.
Minecraft videos only (points.bin game flag 0), by upload year:

  topics by year  per year, the clusters most over-represented vs all years: by uploads (what got made) and by
                  views (what got watched), as share-that-year / share-overall (eras.py prints the same)
  length mix      share of long videos by length bucket
  looks           how common each thumbnail look is per year: linear probes (look_probe.py) trained on
                  vision-model labels; mean probability = share estimate
  engagement      likes per 1,000 views, long vs Short
The older within-channel views comparison lives in analyze_views.py.
  nice uv run python analyze.py
"""
import glob, json, os

import numpy as np

from look_probe import thumb_vecs

ROOT = os.path.dirname(os.path.abspath(__file__))
D, W = os.path.join(ROOT, "data"), os.path.join(ROOT, "web", "data")
PT = [("x", "<f4"), ("y", "<f4"), ("ch", "<u2"), ("m", "<u2"), ("cl", "<i2"), ("kind", "u1"), ("game", "u1")]
YEARS = list(range(2011, 2027))

meta = json.load(open(os.path.join(W, "meta.json")))
r = np.fromfile(os.path.join(W, "points.bin"), dtype=PT)
N = len(r)
views = np.fromfile(os.path.join(W, "views.bin"), dtype="<u4").astype(np.float64)
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
nc = max(names) + 1
out = {"years": YEARS, "n": int(mc.sum())}

# ---- topics by year: over-represented clusters, by uploads and by views
cl = r["cl"]
inc = mc & (cl >= 0)
isMC = np.array([types.get(k) == "mc" for k in range(nc)])
all_up = np.bincount(cl[inc], minlength=nc) / mc.sum()
all_v = np.bincount(cl[inc], weights=views[inc], minlength=nc) / views[mc].sum()
by_year = []
for y in YEARS:
    s = mc & (year == y)
    sc = s & (cl >= 0)
    cnt = np.bincount(cl[sc], minlength=nc)
    chs = np.array([len(np.unique(r["ch"][sc & (cl == k)])) if cnt[k] >= 40 else 0 for k in range(nc)])
    okk = isMC & (cnt >= 40) & (chs >= 3)
    up = cnt / s.sum() / np.maximum(all_up, 1e-12)
    vw = np.bincount(cl[sc], weights=views[sc], minlength=nc) / max(views[s].sum(), 1) / np.maximum(all_v, 1e-12)
    pick = lambda lift: [{"id": int(k), "name": names[k], "x": round(float(lift[k]), 1), "n": int(cnt[k])}
                         for k in np.argsort(-lift) if okk[k]][:3]
    by_year.append({"year": y, "videos": int(s.sum()), "made": pick(up), "watched": pick(vw)})
out["topics_by_year"] = by_year

# ---- length mix of long videos
BUCKETS = [("Under 5 min", 0, 300), ("5–10 min", 300, 600), ("10–20 min", 600, 1200), ("20–40 min", 1200, 2400),
           ("40 min–1 h", 2400, 3600), ("Over 1 h", 3600, 1e9)]
lm = {"buckets": [b[0] for b in BUCKETS], "share": []}
for y in YEARS:
    s = mc & (year == y) & (r["kind"] == 0) & (dur > 60)
    lm["share"].append([round(float(((dur[s] >= a) & (dur[s] < b)).mean()), 4) for _, a, b in BUCKETS] if s.sum() > 200 else None)
out["length_mix"] = lm

# ---- thumbnail looks, how common per year
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
    w, (auc, acc, prec, rec, share, _) = PR[k], PR[k + "__stats"]
    if auc < 0.9:
        continue
    pr = 1 / (1 + np.exp(-(TV @ w[:768] + w[768])))
    tr = lambda kd: [round(float(pr[mc & has & (year == y) & (r["kind"] == kd)].mean()), 4)
                     if (mc & has & (year == y) & (r["kind"] == kd)).sum() > 300 else None for y in YEARS]
    looks.append({"key": k, "name": name, "auc": round(float(auc), 3), "precision": round(float(prec), 2),
                  "trend": [round(float(pr[mc & has & (year == y)].mean()), 4) for y in YEARS], "long": tr(0), "shorts": tr(1)})
out["looks_labelled"] = sum(1 for _ in open(os.path.join(D, "thumb_labels.jsonl")))
out["looks"] = sorted(looks, key=lambda l: -(l["trend"][-1] - l["trend"][0]))

# ---- likes per 1,000 views
eng = {"long": [], "short": []}
for y in YEARS:
    for kd, key in ((0, "long"), (1, "short")):
        s = mc & (year == y) & (r["kind"] == kd) & (views > 1000) & (likes > 0)
        eng[key].append(round(float(np.median(likes[s] / views[s] * 1000)), 1) if s.sum() > 200 else None)
out["engagement"] = eng

json.dump(out, open(os.path.join(W, "analysis.json"), "w"), ensure_ascii=False)
for y in by_year:
    print(y["year"], "made:", [(t["name"], t["x"]) for t in y["made"]], "| watched:", [(t["name"], t["x"]) for t in y["watched"]])
print("LENGTH MIX", lm["buckets"])
for y, s in zip(YEARS, lm["share"]):
    print(" ", y, s)
print("LOOKS (2011 -> 2026 share)")
for l in out["looks"]:
    print(f"  {l['name']:28s} {l['trend'][0]:.0%} -> {l['trend'][-1]:.0%}   peak {YEARS[int(np.argmax(l['trend']))]}  shorts 2026 {l['shorts'][-1]}")
print("ENGAGEMENT", eng)
