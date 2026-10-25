"""The best finds from scan.py, shaped for the blog -> web/data/finds.json (title map, Minecraft videos only).

  adbreak    long-video length in 5 s bins from 5:00 to 12:00, by era, + per year how much more common just-over-10:00 / just-over-8:00 is
  shapes     title shape lines: starts "Minecraft:", ends "in Minecraft", all lowercase, emoji, w/ or ft., median title length
  emoji      top emoji per year;  numbers: most-used number per year (ignoring 1-5);  versions: the 1.X a year talks about most
  weekday    share of uploads per weekday, per year
  mobs       monthly mentions of new mobs from 2019 on (for announce -> release timelines)
  nice uv run python finds.py
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
titles = open(os.path.join(W, "titles.txt"), newline="").read().split("\n")[:N]
ids = open(os.path.join(W, "ids.txt")).read().split("\n")[:N]
year = (r["m"] // 12 + 2005).astype(int)
keep = (r["game"] == 0) & (year >= 2011) & (year <= 2026)
idx = np.flatnonzero(keep)
low = [t.lower() for t in titles]
out = {"years": YEARS}

dur = {}
for p in glob.glob(os.path.join(D, "details", "*.jsonl")):
    for l in open(p):
        d = json.loads(l)
        dur[d["id"]] = d.get("dur") or 0
du = np.array([dur.get(i, 0) for i in ids])
long_ = keep & (r["kind"] == 0)

# ---- ad-break lengths
ab = {"bins": list(range(300, 720, 5)), "eras": {}}
for name, a, b in (("2016–2019", 2016, 2019), ("2021–2026", 2021, 2026)):
    s = du[long_ & (year >= a) & (year <= b)]
    h = np.histogram(s, bins=list(range(300, 725, 5)))[0]
    ab["eras"][name] = (h / max((long_ & (year >= a) & (year <= b)).sum(), 1)).round(6).tolist()
over = lambda s, t: float(((s >= t) & (s < t + 30)).sum() / max(1, ((s >= t - 30) & (s < t)).sum()))
ab["over10"] = [round(over(du[long_ & (year == y)], 600), 2) for y in YEARS]
ab["over8"] = [round(over(du[long_ & (year == y)], 480), 2) for y in YEARS]
out["adbreak"] = ab

# ---- title shapes
ytot = {y: (keep & (year == y)).sum() for y in YEARS}
def ser(hit):
    return [round(float((hit & keep & (year == y)).sum() / ytot[y]), 4) for y in YEARS]
rx = lambda p: np.array([bool(re.search(p, t)) for t in low])
EMO = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
out["shapes"] = {
    "starts": ser(rx(r"^minecraft\s*[:\-|]")),
    "ends": ser(rx(r"in minecraft[\s!?.]*(\(|\[|#|$)")),
    "lower": ser(np.array([t == t.lower() and any(c.isalpha() for c in t) for t in titles])),
    "emoji": ser(np.array([bool(EMO.search(t)) for t in titles])),
    "collab": ser(rx(r"\bw/|\bft\.?\s|\bfeat\.?\s|\bwith\b")),
    "i": ser(rx(r"^i\b|\bi\b")),
    "chars": [float(np.median([len(titles[i]) for i in idx if year[i] == y])) for y in YEARS],
}
ex = {}
for y in (2012, 2025):  # an example title for each shape end
    s = [i for i in idx if year[i] == y]
    ex[y] = next(titles[i] for i in s if re.search(r"^minecraft\s*[:\-|]" if y == 2012 else r"in minecraft$", low[i]) and len(titles[i]) < 60)
out["shape_examples"] = ex

# ---- emoji, numbers, versions per year
emo, nums, vers = {y: Counter() for y in YEARS}, {y: Counter() for y in YEARS}, {y: Counter() for y in YEARS}
num = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?![\w.])")
ver = re.compile(r"\b1\.(\d{1,2})\b")
for i in idx:
    y = int(year[i])
    for e in set(EMO.findall(titles[i])):
        emo[y][e] += 1
    for n in {int(x.replace(",", "")) for x in num.findall(titles[i])}:
        if n > 5 and not 1990 <= n <= 2030:
            nums[y][n] += 1
    for v in set(ver.findall(low[i])):
        if 0 < int(v) <= 22:
            vers[y][int(v)] += 1
out["emoji"] = {y: [[e, c] for e, c in emo[y].most_common(3)] for y in YEARS}
out["numbers"] = {y: [[n, c] for n, c in nums[y].most_common(3)] for y in YEARS}
out["versions"] = {y: {"top": vers[y].most_common(1)[0][0] if vers[y] else None, "share": round(sum(vers[y].values()) / ytot[y], 4)} for y in YEARS}
# what the year's top number is mostly attached to
ctx = {}
for y in YEARS:
    n = nums[y].most_common(1)[0][0]
    c = Counter()
    for i in idx:
        if year[i] == y:
            m = re.search(rf"(\w+)\s+{n}\b|\b{n}\s+(\w+)", low[i])
            if m:
                c[m.group(1) or m.group(2)] += 1
    ctx[y] = c.most_common(3)
out["number_context"] = ctx

# ---- weekday (Pacific)
date = {}
for p in glob.glob(os.path.join(D, "videos", "*.jsonl")):
    for l in open(p):
        i, j = l.find('"date": "'), l.find('"id": "')
        date[l[j + 7:l.index('"', j + 7)]] = l[i + 9:i + 29]
PST = timezone(timedelta(hours=-8))
wd = defaultdict(lambda: np.zeros(7))
mon = {}
for i in idx:
    d = date.get(ids[i])
    if d:
        t = datetime.strptime(d, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).astimezone(PST)
        wd[t.year][t.weekday()] += 1
        mon[i] = (t.year - 2019) * 12 + t.month - 1
out["weekday"] = {y: (wd[y] / wd[y].sum()).round(4).tolist() for y in YEARS}

# ---- new mobs, monthly from Jan 2019
MOBS = {"Piglin": r"piglin", "Axolotl": r"axolotl", "Warden": r"warden", "Allay": r"\ballay", "Sniffer": r"sniffer", "Camel": r"\bcamels?\b",
        "Armadillo": r"armadillo", "Breeze": r"\bbreeze", "Creaking": r"creaking", "Happy ghast": r"happy ghast"}
M = (2026 - 2019) * 12 + 10
mt = np.bincount([mon[i] for i in mon if mon[i] >= 0], minlength=M)[:M]
mobs = {}
for k, p in MOBS.items():
    c = re.compile(p)
    h = np.bincount([mon[i] for i in mon if mon[i] >= 0 and c.search(low[i])], minlength=M)[:M]
    mobs[k] = {"count": h.tolist(), "share": (h / np.maximum(mt, 1)).round(5).tolist(),
               "first": int(np.argmax(h >= 3)) if (h >= 3).any() else None}
out["mobs"] = mobs
out["mobs_m0"] = "2019-01"

json.dump(out, open(os.path.join(W, "finds.json"), "w"), ensure_ascii=False)
print("over10", ab["over10"]); print("over8 ", ab["over8"])
print("examples", ex)
print("numbers", {y: v[0] for y, v in out["numbers"].items()})
print("number ctx", ctx)
print("versions", {y: v["top"] for y, v in out["versions"].items()})
for k, v in mobs.items():
    f = v["first"]
    print(f"  {k:12s} first 3+/month: {2019 + f // 12}-{f % 12 + 1:02d}" if f is not None else k, " peak", 2019 + int(np.argmax(v["count"])) // 12, int(np.argmax(v["count"])) % 12 + 1)
