"""Patterns for the blog -> web/data/stats.json (title map, Minecraft videos only unless noted).

  river       share of each year's uploads in the biggest topics (each year's top 3 by share, merged)
  lifespans   per topic that took off: months its share stayed above half its peak, vs the peak date
  words       per year, the title words/phrases that rose most vs the two years before (5+ channels)
  keywords    share of titles naming a few famous things (herobrine, dream, hardcore, ...)
  survival    of the channels active in year y, the share still posting k years later
  breadth     how many topics it takes to cover half of a year's clustered uploads
  timing      share uploaded on weekends; median upload hour (US Pacific)
  top_video   the most-viewed video uploaded each year
  nice uv run python stats.py
"""
import glob, json, os, re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
D, W = os.path.join(ROOT, "data"), os.path.join(ROOT, "web", "data")
PT = [("x", "<f4"), ("y", "<f4"), ("ch", "<u2"), ("m", "<u2"), ("cl", "<i2"), ("kind", "u1"), ("game", "u1")]
YEARS = list(range(2011, 2027))

meta = json.load(open(os.path.join(W, "meta.json")))
r = np.fromfile(os.path.join(W, "points.bin"), dtype=PT)
N = len(r)
views = np.fromfile(os.path.join(W, "views.bin"), dtype="<u4").astype(np.float64)
titles = open(os.path.join(W, "titles.txt"), newline="").read().split("\n")[:N]
ids = open(os.path.join(W, "ids.txt")).read().split("\n")[:N]
names = {c["id"]: c["name"] for c in meta["clusters"]}
types = {c["id"]: c["type"] for c in meta["clusters"]}
chname = [c["name"] for c in meta["channels"]]
year, mon = r["m"] // 12 + 2005, r["m"].astype(int)
mc = r["game"] == 0
cl = r["cl"]
out = {"years": YEARS, "n": int(mc.sum())}

# ---- river
ytot = {y: (mc & (year == y)).sum() for y in YEARS}
share = {k: np.array([(mc & (year == y) & (cl == k)).sum() / ytot[y] for y in YEARS]) for k in names if types[k] == "mc"}
pick = []
for i, y in enumerate(YEARS):
    for k in sorted(share, key=lambda k: -share[k][i])[:3]:
        if k not in pick:
            pick.append(k)
pick.sort(key=lambda k: int(np.argmax(share[k])) + share[k].argmax() / 100)
out["river"] = {"topics": [{"id": int(k), "name": names[k], "share": [round(float(v), 4) for v in share[k]],
                            "peak": YEARS[int(np.argmax(share[k]))]} for k in pick]}

# ---- lifespans: monthly share, 3-month smoothed, months at >= half the peak
M0, M1 = (2010 - 2005) * 12, (2026 - 2005) * 12 + 9
mt = np.bincount(mon[mc], minlength=M1 + 1).astype(float)
life = []
for k in names:
    if types[k] != "mc":
        continue
    sel = mc & (cl == k)
    if sel.sum() < 250 or len(np.unique(r["ch"][sel])) < 5:
        continue
    s = np.bincount(mon[sel], minlength=M1 + 1)[M0:M1] / np.maximum(mt[M0:M1], 1)
    s = np.convolve(s, np.ones(3) / 3, "same")
    pk = int(np.argmax(s))
    overall = sel.sum() / mc.sum()
    if s[pk] < 3 * overall:
        continue  # a steady genre, not a trend
    lo = pk
    while lo > 0 and s[lo - 1] >= s[pk] / 2:
        lo -= 1
    hi = pk
    while hi < len(s) - 1 and s[hi + 1] >= s[pk] / 2:
        hi += 1
    m = M0 + pk
    life.append({"id": int(k), "name": names[k], "peak": round(2005 + m / 12, 2), "months": hi - lo + 1, "n": int(sel.sum()),
                 "ongoing": hi == len(s) - 1})
out["lifespans"] = life
done = [l for l in life if not l["ongoing"]]
by_era = {}
for a, b in ((2010, 2015), (2015, 2020), (2020, 2027)):
    v = [l["months"] for l in done if a <= l["peak"] < b]
    by_era[f"{a}–{min(b, 2026) - (0 if b == 2027 else 1)}"] = {"median": float(np.median(v)) if v else None, "n": len(v)}
out["lifespan_by_era"] = by_era

# ---- words that rose each year
STOP = set("the a an of in on to and i my is for with it you this we me at be by from or are so but not all how what our ep part episode "
           "minecraft mc shorts short video vs new".split())
tok = lambda t: re.findall(r"[a-z0-9']+", t.lower())
cnt = {y: Counter() for y in YEARS}
chs = {y: defaultdict(set) for y in YEARS}
for i in np.flatnonzero(mc):
    y = int(year[i])
    if y not in cnt:
        continue
    w = tok(titles[i])
    good = lambda g: g not in STOP and len(g) > 1 and not (g.isdigit() and len(g) < 3)  # "100" ok, "16" or "x" not
    grams = {g for g in w if good(g) and len(g) > 2 and not g.isdigit()} | {f"{a} {b}" for a, b in zip(w, w[1:]) if good(a) and good(b)}
    for g in grams:
        cnt[y][g] += 1
        if len(chs[y][g]) < 6:
            chs[y][g].add(int(r["ch"][i]))
words = []
for y in YEARS[2:]:
    prev = cnt[y - 1] + cnt[y - 2]
    pt = ytot[y - 1] + ytot[y - 2]
    sc = []
    for g, c in cnt[y].items():
        if c < 40 or len(chs[y][g]) < 5:
            continue
        a, b = c / ytot[y], (prev[g] + 2) / pt
        sc.append((a / b, g, a))
    sc.sort(reverse=True)
    top = []
    for x, g, a in sc:  # skip a bigram whose word is already in, and vice versa
        if any(g in t or t in g for t, _, _ in top):
            continue
        top.append((g, round(float(x), 1), round(float(a), 4)))
        if len(top) == 6:
            break
    words.append({"year": y, "words": [{"w": g, "x": x, "share": a} for g, x, a in top]})
out["words"] = words

# ---- keywords
KW = {"Herobrine": r"herobrine", "Dream": r"\bdream\b", "Hardcore": r"hardcore", "SMP": r"\bsmp\b", "100 days": r"100 days",
      "Hypixel": r"hypixel", "Skyblock": r"sky ?block", "Mods": r"\bmods?\b", "Speedrun": r"speedrun", "Hunger Games": r"hunger games",
      "Mod pack / modpack": r"mod ?pack|ftb|tekkit", "Seed": r"\bseeds?\b", "Parkour": r"parkour", "Herobrine & Entity 303": r"entity 303",
      "Prank / troll": r"\bprank|\btroll", "Noob vs Pro": r"noob", "Girlfriend": r"girlfriend", "$": r"\$"}
kw = {}
T = np.array(titles, dtype=object)
for name, pat in KW.items():
    rx = re.compile(pat, re.I)
    hit = np.array([bool(rx.search(t)) for t in titles])
    kw[name] = [round(float((hit & mc & (year == y)).sum() / ytot[y]), 4) for y in YEARS]
out["keywords"] = kw

# ---- channel survival: active = 5+ Minecraft uploads that year
act = np.zeros((len(chname), len(YEARS)), bool)
for j, y in enumerate(YEARS):
    c = np.bincount(r["ch"][mc & (year == y)], minlength=len(chname))
    act[:, j] = c >= 5
surv = {}
for j, y in enumerate(YEARS[:-1]):
    base = act[:, j]
    if base.sum() < 20:
        continue
    surv[y] = {"channels": int(base.sum()), "still": [round(float(act[base, j + k].mean()), 3) for k in range(1, len(YEARS) - j)]}
out["survival"] = surv
newch = [int((act[:, j] & ~act[:, :j].any(1)).sum()) for j in range(len(YEARS))]
out["new_channels"] = newch
out["active_channels"] = [int(act[:, j].sum()) for j in range(len(YEARS))]

# ---- breadth: topics needed to cover half the clustered uploads
br = []
for y in YEARS:
    c = np.bincount(cl[mc & (year == y) & (cl >= 0)])
    c = np.sort(c)[::-1]
    br.append(int(np.searchsorted(np.cumsum(c), c.sum() / 2) + 1))
out["breadth"] = br
out["clustered"] = [round(float((mc & (year == y) & (cl >= 0)).sum() / ytot[y]), 3) for y in YEARS]

# ---- timing (dates carry a UTC time)
date = {}
for p in glob.glob(os.path.join(D, "videos", "*.jsonl")):
    for l in open(p):
        i = l.find('"date": "')
        j = l.find('"id": "')
        date[l[j + 7:l.index('"', j + 7)]] = l[i + 9:i + 29]
PST = timezone(timedelta(hours=-8))
wk, hr = defaultdict(list), defaultdict(list)
for i in np.flatnonzero(mc):
    d = date.get(ids[i])
    if not d:
        continue
    t = datetime.strptime(d, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).astimezone(PST)
    wk[t.year].append(t.weekday() >= 5)
    hr[t.year].append(t.hour + t.minute / 60)
out["weekend"] = [round(float(np.mean(wk[y])), 3) for y in YEARS]
out["hour"] = [round(float(np.median(hr[y])), 1) for y in YEARS]
out["hours_hist"] = {y: np.histogram(hr[y], bins=24, range=(0, 24))[0].tolist() for y in (2012, 2018, 2025)}

# ---- most viewed per year
tv = []
for y in YEARS:
    s = np.flatnonzero(mc & (year == y))
    i = s[np.argmax(views[s])]
    tv.append({"year": y, "title": titles[i], "channel": chname[r["ch"][i]], "views": int(views[i]), "id": ids[i],
               "short": int(r["kind"][i]), "cl": names.get(int(cl[i]), "")})
out["top_video"] = tv

json.dump(out, open(os.path.join(W, "stats.json"), "w"), ensure_ascii=False)
print("RIVER", [(t["name"], t["peak"], max(t["share"])) for t in out["river"]["topics"]])
print("LIFESPAN BY ERA", by_era)
print("LIFE", sorted([(l["peak"], l["months"], l["name"]) for l in life]))
for w in words:
    print(w["year"], [(x["w"], x["x"]) for x in w["words"]])
for k, v in kw.items():
    print(f"  {k:24s}", " ".join(f"{x:.3f}" for x in v))
print("SURVIVAL", {y: (s["channels"], s["still"][:3], s["still"][-1]) for y, s in surv.items()})
print("NEW CH", newch, "\nACTIVE", out["active_channels"])
print("BREADTH", br, "\nCLUSTERED", out["clustered"])
print("WEEKEND", out["weekend"], "\nHOUR", out["hour"])
for t in tv:
    print(t["year"], fmt := f"{t['views']:,}", t["channel"], "|", t["title"], "|", t["cl"])
