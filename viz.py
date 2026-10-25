"""Data for the blog's picture sections -> web/img/*.png|jpg + web/data/viz.json (title map, Minecraft videos only).

  avg thumbs   the pixel average of up to 3,000 random long-video thumbnails per year -> web/img/avg_YYYY.jpg
  year maps    the title map, 12k random videos of one year in mint over all years in grey -> web/img/map_YYYY.png
  spread       for a few trends, the month each channel first made 2+ videos in that topic
  lengths      per year, how many videos fall in each log-spaced length bin, long and Shorts apart
  nice uv run python viz.py
"""
import glob, json, os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
D, W = os.path.join(ROOT, "data"), os.path.join(ROOT, "web", "data")
IMG = os.path.join(ROOT, "web", "img")
os.makedirs(IMG, exist_ok=True)
PT = [("x", "<f4"), ("y", "<f4"), ("ch", "<u2"), ("m", "<u2"), ("cl", "<i2"), ("kind", "u1"), ("game", "u1")]
YEARS = list(range(2011, 2027))

meta = json.load(open(os.path.join(W, "meta.json")))
r = np.fromfile(os.path.join(W, "points.bin"), dtype=PT)
N = len(r)
ids = open(os.path.join(W, "ids.txt")).read().split("\n")[:N]
chname = [c["name"] for c in meta["channels"]]
names = {c["id"]: c["name"] for c in meta["clusters"]}
year = r["m"] // 12 + 2005
mc = r["game"] == 0
out = {"years": YEARS}
rng = np.random.default_rng(0)

# ---- average thumbnail per year (long videos; Shorts thumbs are letterboxed)
avg_n = {}
for y in YEARS:
    s = np.flatnonzero(mc & (year == y) & (r["kind"] == 0))
    rng.shuffle(s)
    acc, n = np.zeros((180, 320, 3)), 0
    for i in s:
        p = os.path.join(D, "thumbs", ids[i][:2], ids[i] + ".jpg")
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert("RGB")
        if im.size != (320, 180):
            im = im.resize((320, 180))
        acc += np.asarray(im, np.float64)
        n += 1
        if n == 3000:
            break
    a = acc / max(n, 1)
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)  # averages come out flat grey-brown, so stretch the contrast
    Image.fromarray((np.clip((a - lo) / (hi - lo), 0, 1) * 255).round().astype(np.uint8)).save(os.path.join(IMG, f"avg_{y}.jpg"), quality=90)
    avg_n[y] = n
out["avg_thumbs"] = avg_n

# ---- the map, one year lit up: 12,000 random videos from that year in mint over every year in grey
x0, x1 = np.quantile(r["x"][mc], [0.035, 0.97])  # crop to the main continent; the stray islands sit far out
y0, y1 = np.quantile(r["y"][mc], [0.035, 0.97])
c, hw = ((x0 + x1) / 2, (y0 + y1) / 2), max(x1 - x0, y1 - y0) * 0.56
x0, x1, y0, y1 = c[0] - hw, c[0] + hw, c[1] - hw, c[1] + hw
S = 560 * 2  # draw at 2x, then shrink for soft dots
gx = ((r["x"] - x0) / (x1 - x0) * S).astype(int)
gy = ((y1 - r["y"]) / (y1 - y0) * S).astype(int)
inb = (gx >= 0) & (gx < S) & (gy >= 0) & (gy < S)
def splat(sel):
    h = np.zeros((S, S))
    np.add.at(h, (gy[sel & inb], gx[sel & inb]), 1)
    return h
allh = splat(mc)
from scipy.ndimage import gaussian_filter
g = 1 - np.exp(-gaussian_filter(allh, 1.0) * 0.35)
ground, grey, mint = np.array([11, 13, 17]), np.array([70, 76, 88]), np.array([201, 245, 214])
for y in YEARS:
    s = np.flatnonzero(mc & (year == y))
    pick = np.zeros(N, bool)
    pick[rng.choice(s, min(12000, len(s)), replace=False)] = True
    v = 1 - np.exp(-gaussian_filter(splat(pick), 1.4) * 9)
    rgb = ground + (grey - ground) * g[..., None]
    rgb = rgb * (1 - v[..., None]) + mint * v[..., None]
    Image.fromarray(np.clip(rgb, 0, 255).round().astype(np.uint8)).resize((S // 2, S // 2), Image.LANCZOS).save(os.path.join(IMG, f"map_{y}.png"))

# where each big topic sits on these pictures, 0-1 from the top left, for labels
out["map_labels"] = {}
for k, nm in names.items():
    sel = mc & (r["cl"] == k) & inb
    if sel.sum() >= 400:
        out["map_labels"][int(k)] = {"name": nm, "x": round(float(np.median(gx[sel])) / S, 4), "y": round(float(np.median(gy[sel])) / S, 4)}

# ---- how a trend spreads through channels
TREND = [106, 102, 26, 75, 55, 153]  # Hunger Games, Bed Wars, Noob vs Pro, I Survived 100 Days as, Manhunt but, Lifesteal SMP
M0 = (2011 - 2005) * 12
mt = np.bincount(r["m"][mc], minlength=(2027 - 2005) * 12).astype(float)
spread = []
for k in TREND:
    sel = mc & (r["cl"] == k)
    dots = []
    for c in np.unique(r["ch"][sel]):
        m = np.sort(r["m"][sel & (r["ch"] == c)])
        if len(m) >= 2:
            dots.append({"ch": chname[c], "m": int(m[1]) - M0, "n": int(len(m))})  # 2nd video = actually joined in
    dots.sort(key=lambda d: d["m"])
    share = np.bincount(r["m"][sel], minlength=len(mt)) / np.maximum(mt, 1)
    spread.append({"id": k, "name": names[k], "dots": dots, "share": [round(float(v), 4) for v in share[M0:]]})
out["spread"] = spread
out["spread_m0"] = "2011-01"

# ---- video length distribution per year
dur = {}
for p in glob.glob(os.path.join(D, "details", "*.jsonl")):
    for l in open(p):
        d = json.loads(l)
        dur[d["id"]] = d.get("dur") or 0
du = np.array([dur.get(i, 0) for i in ids], float)
edges = np.logspace(np.log10(10), np.log10(4 * 3600), 61)
ln = {"edges": [round(float(e), 1) for e in edges], "long": [], "short": []}
for y in YEARS:
    for kd, key in ((0, "long"), (1, "short")):
        s = mc & (year == y) & (r["kind"] == kd) & (du > 0)
        ln[key].append(np.histogram(du[s], edges)[0].tolist())
out["lengths"] = ln

json.dump(out, open(os.path.join(W, "viz.json"), "w"), ensure_ascii=False)
print("AVG THUMBS", avg_n)
for t in spread:
    print(t["name"], len(t["dots"]), "channels, first", t["dots"][0]["ch"], t["dots"][0]["m"] // 12 + 2011)
