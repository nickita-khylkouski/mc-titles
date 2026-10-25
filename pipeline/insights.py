"""Numbers for the UI's Insights panel -> web/data{,_rich,_thumb}/insights.json (MAP_SRC=title|rich|thumb).

Minecraft-only videos (points.bin game flag 0), by upload year:
  drift      1 - cos(mean vector of year y, mean vector of year y-1): how much the average video changed
  spread     1 - |mean of unit vectors|^2 = average cosine distance between two random videos that year
  newness    for a sample of each year's videos, the similarity of its closest video from any EARLIER year;
             reported as the share with nothing close before (sim < NEW) and the median best similarity
  starters   per cluster that took off: the first channels in it (by the date of their 3rd video there)
  honesty    for 2,000 random videos, where their 12 true nearest neighbours landed on the 2-d map
Title map only (the same for every map, so stored once): title style, video length, concentration, Shorts views.
  MAP_SRC=rich nice uv run python pipeline/insights.py
"""
import glob, json, os, re
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
SRC = os.environ.get("MAP_SRC", "title")
SFX = "" if SRC == "title" else "_" + SRC
W = os.path.join(ROOT, "web", "data" + SFX)
NEW = {"title": 0.6, "rich": 0.6, "thumb": 0.82}[SRC]  # "nothing like it before": ~median - 0.15 for text; image vectors sit ~2x tighter
rng = np.random.default_rng(0)

meta = json.load(open(os.path.join(W, "meta.json")))
r = np.fromfile(os.path.join(W, "points.bin"), dtype=[("x", "<f4"), ("y", "<f4"), ("ch", "<u2"), ("m", "<u2"),
                                                      ("cl", "<i2"), ("kind", "u1"), ("game", "u1")])
V = np.load(os.path.join(D, f"vecs{SFX}.npy")).astype(np.float32)
assert len(V) == len(r)
views = np.fromfile(os.path.join(W, "views.bin"), dtype="<u4").astype(np.float64)
titles = open(os.path.join(W, "titles.txt"), newline="").read().split("\n")[:len(r)]
ids = open(os.path.join(W, "ids.txt")).read().split("\n")[:len(r)]
mc = r["game"] == 0
year = r["m"] // 12 + 2005
YEARS = [y for y in range(2010, 2027) if (mc & (year == y)).sum() >= 1000]
out = {"years": YEARS, "src": SRC}

# ---- drift / spread
means = {y: V[mc & (year == y)].mean(0) for y in YEARS}
out["drift"] = [None] + [round(float(1 - means[a] @ means[b] / np.linalg.norm(means[a]) / np.linalg.norm(means[b])), 4)
                         for a, b in zip(YEARS, YEARS[1:])]
out["spread"] = [round(float(1 - means[y] @ means[y]), 4) for y in YEARS]
first = YEARS[0]
out["from_first"] = [round(float(1 - means[y] @ means[first] / np.linalg.norm(means[y]) / np.linalg.norm(means[first])), 4) for y in YEARS]

# ---- newness: best similarity to anything uploaded in an earlier year
new_share, new_med, new_ex = [None], [None], [None]
for y in YEARS[1:]:
    cur = np.flatnonzero(mc & (year == y))
    prev = np.flatnonzero(mc & (year < y))
    smp = rng.choice(cur, min(800, len(cur)), replace=False)
    best = np.full(len(smp), -1.0, dtype=np.float32)
    for s in range(0, len(prev), 40000):
        best = np.maximum(best, (V[prev[s:s + 40000]] @ V[smp].T).max(0))
    new_share.append(round(float((best < NEW).mean()), 4))
    new_med.append(round(float(np.median(best)), 4))
    newest = smp[np.argsort(best)[:6]]  # examples: the year's videos least like anything before
    new_ex.append([titles[i] for i in newest])
out["new_share"], out["new_median"], out["new_examples"], out["new_threshold"] = new_share, new_med, new_ex, NEW

# ---- trend starters: clusters with a clear peak year; first channels in by their 3rd video there
names = {c["id"]: c["name"] for c in meta["clusters"]}
types = {c["id"]: c["type"] for c in meta["clusters"]}
ytot = {y: (mc & (year == y)).sum() for y in YEARS}
starters = []
for c in meta["clusters"]:
    k = c["id"]
    idx = np.flatnonzero(mc & (r["cl"] == k))
    if len(idx) < 250 or types[k] != "mc":
        continue
    share = np.array([((year[idx] == y).sum()) / ytot[y] for y in YEARS])
    pk = int(np.argmax(share))
    allshare = len(idx) / mc.sum()
    if share[pk] < 3 * allshare:
        continue  # no real peak, a steady genre
    by = defaultdict(list)
    for i in idx[np.argsort(r["m"][idx], kind="stable")]:
        by[int(r["ch"][i])].append(int(r["m"][i]))
    third = sorted((ms[2], ch) for ch, ms in by.items() if len(ms) >= 3)
    if len(third) < 4:
        continue
    joined_before_peak = sum(1 for m, _ in third if m < (YEARS[pk] - 2005) * 12)
    starters.append({"id": k, "name": names[k], "n": int(len(idx)), "peak": YEARS[pk], "lift": round(float(share[pk] / allshare), 1),
                     "first": [[meta["channels"][ch]["name"], m] for m, ch in third[:4]],
                     "channels": len(third), "before_peak": joined_before_peak,
                     "lead": round((YEARS[pk] - 2005) - third[0][0] / 12, 1)})
starters.sort(key=lambda s: -s["lift"] * np.log(s["n"]))
out["starters"] = starters[:40]

# ---- honesty of the 2-d map
xy = np.stack([r["x"], r["y"]], 1)
smp = rng.choice(len(r), 2000, replace=False)
near, med_rank, kept_cluster = [], [], []
for s0 in range(0, len(smp), 250):
    ss = smp[s0:s0 + 250]
    sims = V @ V[ss].T
    sims[ss, np.arange(len(ss))] = -9
    top = np.argpartition(-sims, 12, axis=0)[:12].T  # (250, 12)
    for j, i in enumerate(ss):
        d2 = ((xy - xy[i]) ** 2).sum(1)
        srt = np.sort(d2)
        rk = np.searchsorted(srt, d2[top[j]])
        near.append((rk < len(r) // 100).mean())
        med_rank.append(np.median(rk) / len(r))
        if r["cl"][i] >= 0:
            kept_cluster.append((r["cl"][top[j]] == r["cl"][i]).mean())
# a random baseline: for random pairs, how often they're in each other's closest 1% (= 1% by construction)
out["honesty"] = {"near1": round(float(np.mean(near)), 3), "median_rank_pct": round(float(np.median(med_rank)) * 100, 2),
                  "same_cluster": round(float(np.mean(kept_cluster)), 3), "k": 12, "sample": 2000, "dims": int(V.shape[1])}

# ---- title-map-only stats (same videos/titles in every map)
if SRC == "title":
    EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
    style = defaultdict(list)
    for y in YEARS:
        T = [titles[i] for i in np.flatnonzero(mc & (year == y))]
        words = [t.split() for t in T]
        caps = [sum(w.isupper() and len(w) > 1 for w in ws) / max(1, sum(len(w) > 1 for w in ws)) for ws in words]
        style["words"].append(round(float(np.median([len(w) for w in words])), 1))
        style["caps"].append(round(float(np.mean([c >= 0.5 for c in caps])), 4))
        style["emoji"].append(round(float(np.mean([bool(EMOJI.search(t)) for t in T])), 4))
        style["question"].append(round(float(np.mean(["?" in t for t in T])), 4))
        style["first_person"].append(round(float(np.mean([bool(re.match(r"(i|i'm|i've|we|my)\b", t.strip().lower())) for t in T])), 4))
        style["vs"].append(round(float(np.mean([bool(re.search(r"\bvs\.?\b", t.lower())) for t in T])), 4))
        style["episode"].append(round(float(np.mean([bool(re.search(r"(\bep\.?\s*\d|episode\s*\d|#\d|\bpart\s*\d|\bday\s*\d)", t.lower())) for t in T])), 4))
        style["money"].append(round(float(np.mean(["$" in t for t in T])), 4))
        style["hundred"].append(round(float(np.mean([bool(re.search(r"\b(100|1000|1,000)\b", t)) for t in T])), 4))
    out["style"] = style
    dur = {}
    for p in glob.glob(os.path.join(D, "details", "*.jsonl")):
        for d in map(json.loads, open(p)):
            dur[d["id"]] = d["dur"]
    du = np.array([dur.get(i, 0) for i in ids], dtype=np.float64)
    conc = defaultdict(list)
    for y in YEARS:
        sel = mc & (year == y)
        lng = sel & (r["kind"] == 0) & (du > 60)
        conc["long_minutes"].append(round(float(np.median(du[lng]) / 60), 1) if lng.sum() > 200 else None)
        conc["shorts"].append(round(float((r["kind"][sel] == 1).mean()), 4))
        cnt = np.bincount(r["ch"][sel], minlength=len(meta["channels"]))
        conc["active"].append(int((cnt >= 5).sum()))
        conc["top10"].append(round(float(np.sort(cnt)[-10:].sum() / cnt.sum()), 4))
        kv = sel & (views > 0)
        vv = np.bincount(r["ch"][kv], weights=views[kv], minlength=len(meta["channels"]))
        conc["top10_views"].append(round(float(np.sort(vv)[-10:].sum() / max(1, vv.sum())), 4))
        conc["shorts_views"].append(round(float(views[kv & (r["kind"] == 1)].sum() / max(1, views[kv].sum())), 4))
        conc["median_views_long"].append(int(np.median(views[kv & (r["kind"] == 0)])) if (kv & (r["kind"] == 0)).sum() > 200 else None)
        conc["median_views_short"].append(int(np.median(views[kv & (r["kind"] == 1)])) if (kv & (r["kind"] == 1)).sum() > 200 else None)
    conc["views_coverage"] = round(float((views[mc] > 0).mean()), 3)
    out["conc"] = conc

json.dump(out, open(os.path.join(W, "insights.json"), "w"), ensure_ascii=False)
h = out["honesty"]
print(f"{SRC}: drift {out['drift']}\n spread {out['spread']}\n new {out['new_share']}\n honesty {h}\n "
      f"starters {[(s['name'], s['peak'], s['first'][0][0]) for s in starters[:8]]}")
