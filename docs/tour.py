"""docs/tour.gif: a camera tour of the title map. It starts on all 154 topics, then flies into one topic at a time,
lights it up and shows one of its big videos.  nice uv run python docs/tour.py"""
import json, os, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "tour.gif")
PT = [("x", "<f4"), ("y", "<f4"), ("ch", "<u2"), ("m", "<u2"), ("cl", "<i2"), ("kind", "u1"), ("game", "u1")]
W, H, SS, BAND = 400, 440, 2, 100           # output size, supersampling, label band at the bottom
FPS, INTRO, MOVE, HOLD, ZOOM = 15, 26, 10, 24, 3.0  # short flights: every moving frame costs ~100 KB
DIM = 0.22                                   # brightness of the topics that aren't lit
TOUR = [  # topic, and one of its big videos (the most watched, unless that one is off-topic)
    ("I Survived 100 Days as…", "I Survived 100 Days Of Hardcore Minecraft!"),
    ("Mob battles", "1,000 Tanks Vs Titan Enderman!"),
    ("Updates & snapshots", "Official Minecraft Trailer"),
    ("Bed Wars", "Minecraft bedwars"),
    ("Hermitcraft episodes", "Hermitcraft 7: Episode 1 - HERE WE GO!"),
    ("Hunger Games", "Minecraft Xbox - Disney Pixar - Hunger Games"),
    ("Fooling friends with mods", "50 Ways To Mess With Your Friends In Minecraft"),
    ("FNAF roleplay", '"Join Us For A Bite" | FNAF Sister Location (Animated Minecraft Music Video)'),
]
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
big, small = ImageFont.truetype(FONT, 28 * SS, index=1), ImageFont.truetype(FONT, 15 * SS)

d = os.path.join(ROOT, "web", "data")
r = np.fromfile(os.path.join(d, "points.bin"), dtype=PT)
views = np.fromfile(os.path.join(d, "views.bin"), dtype="<u4")
titles = open(os.path.join(d, "titles.txt")).read().split("\n")
meta = json.load(open(os.path.join(d, "meta.json")))
mc = np.where(r["game"] == 0)[0]  # Minecraft videos only
byname = {c["name"]: c for c in meta["clusters"]}
nshown = sum(1 for c in meta["clusters"] if c.get("type") != "game")
xy = np.c_[r["x"], r["y"]][mc].astype(np.float64)
cl = r["cl"][mc].astype(np.int64)
lo = np.quantile(xy, [0.03, 0.005], 0).diagonal()  # the main body (plus the Hermitcraft island below it)
hi = np.quantile(xy, 0.985, 0)
mid = (lo + hi) / 2
sc = min((W - 16) / (hi[0] - lo[0]), (H - BAND - 16) / (hi[1] - lo[1]))  # pixels per map unit at zoom 1
scr = np.array([W / 2, 8 + (H - BAND - 16) / 2])  # where the camera centre lands on screen (y down)


def cluster_rgb(k):  # same as clusterRGB() in web/index.html; unclustered = grey
    h = (k * 0.61803) % 1
    c = np.clip(np.abs(((h[:, None] * 6 + np.array([0, 4, 2])) % 6) - 3) - 1, 0, 1)
    rgb = 0.5 + (c - 0.5) * 0.85
    rgb[k < 0] = 0.35
    return rgb


col = cluster_rgb(cl)
stops = []
for name, title in TOUR:
    c = byname[name]
    m = cl == c["id"]
    p = xy[m]
    med = np.median(p, 0)
    core = p[np.sqrt(((p - med) ** 2).sum(1)) <= np.quantile(np.sqrt(((p - med) ** 2).sum(1)), 0.5)]
    i = max((i for i in mc[m] if titles[i] == title), key=lambda i: views[i])  # the title has to really be in this topic
    stops.append(dict(name=name, m=m, size=c["size"], at=core.mean(0), rgb=np.median(col[m], 0), title=title, views=int(views[i])))
    print(f"{name}: {c['size']:,} videos, '{title}' {views[i]:,} views", flush=True)


def ease(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def clamp(at, z):  # keep the view over the map instead of flying off into black
    half = np.array([W - 16, H - BAND - 16]) / 2 / (sc * z)
    return np.clip(at, np.minimum(lo + half, mid), np.maximum(hi - half, mid))


def splat(ix, iy, w, sp):
    n = H * SS * W * SS
    out = np.zeros((n, 3))
    for dx in range(sp):
        for dy in range(sp):
            ok = (ix + dx >= 0) & (ix + dx < W * SS) & (iy + dy >= 0) & (iy + dy < H * SS)
            f = (iy[ok] + dy) * W * SS + ix[ok] + dx
            for ch in range(3):
                out[:, ch] += np.bincount(f, w[ok, ch], n)
    return out.reshape(H * SS, W * SS, 3)


def render(cam, z, lit, labels):
    """cam = map point at the screen centre, z = zoom, lit = per-video brightness, labels = [(alpha, title, sub)]"""
    px = ((xy - cam) * sc * z * np.array([1, -1]) + scr) * SS
    ix, iy = px[:, 0].astype(np.int64), px[:, 1].astype(np.int64)
    sp = 2 if z < 1.5 else 3
    cnt = splat(ix, iy, np.ones((len(xy), 3)), sp)[..., :1]
    img = splat(ix, iy, col * lit[:, None], sp) / np.maximum(cnt, 1e-9) * (1 - np.exp(-cnt * 0.8))  # average colour x brightness, by coverage: keeps hues, dim stays dim
    on = lit > DIM + 1e-3
    if on.mean() < 0.5:  # one topic lit: a soft halo in its colour
        hot = splat(ix[on], iy[on], col[on] * (lit[on, None] - DIM), sp)
        img += gaussian_filter(1 - np.exp(-hot * 0.9), (6 * SS, 6 * SS, 0)) * 1.6
    band = np.clip((np.arange(H * SS)[:, None, None] - (H - BAND - 10) * SS) / (24 * SS), 0, 1)
    img *= 1 - 0.85 * band  # fade the map out under the label
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    for a, title, *sub in labels:
        if a < 0.02:
            continue
        dr.text((16 * SS, (H - 62) * SS), title, font=big, fill=(int(245 * a),) * 3, anchor="ls")
        for j, t in enumerate(sub):
            dr.text((16 * SS, (H - 38 + 20 * j) * SS), t, font=small, fill=(int((205 if j == 0 else 130) * a),) * 3, anchor="ls")
    return im.resize((W, H), Image.BOX)


def fit(s, font, width):  # cut a title down to fit with "…"
    if font.getlength(s) <= width:
        return s
    while font.getlength(s + "…") > width:
        s = s[:-1]
    return s.rstrip(" -|:,") + "…"


def label(s):
    v = s["views"]
    t = fit("“" + s["title"].split(" (")[0].replace(chr(34), ""), small, (W - 32) * SS - small.getlength("”")) + "”"
    return s["name"], t, f"{v / 1e6:.0f}M views  ·  one of {s['size']:,} videos"


home = dict(at=mid, z=1.0, lit=np.ones(len(xy)), lab=(f"{nshown} topics", f"{len(xy):,} Minecraft videos, mapped by title", "every colour is one topic, named by hand"))
views_ = [home] + [dict(at=None, z=ZOOM, lit=np.where(s["m"], 1.0, DIM), lab=label(s)) for s in stops] + [home]
for v, s in zip(views_[1:-1], stops):
    v["at"] = clamp(s["at"], ZOOM)

tmp = tempfile.mkdtemp()
n = 0
def save(im):
    global n
    im.save(os.path.join(tmp, f"{n:04d}.png")); n += 1

for k in range(len(views_) - 1):
    a, b = views_[k], views_[k + 1]
    for f in range(INTRO if k == 0 else HOLD):
        if f == 0:
            still = render(a["at"], a["z"], a["lit"], [(1, *a["lab"])])
        save(still)  # held dead still: still frames cost almost nothing in a GIF
    za = a["z"]
    for f in range(1, MOVE + 1):
        u = ease(f / MOVE)
        dip = 0.3 * np.sin(np.pi * f / MOVE) if a["z"] > 1 and b["z"] > 1 else 0  # pull back a little between topics
        z = np.exp(np.log(za) * (1 - u) + np.log(b["z"]) * u - dip)
        at = a["at"] * (1 - u) + b["at"] * u
        lit = a["lit"] * (1 - u) + b["lit"] * u
        save(render(at, z, lit, [(max(0, 1 - 2 * f / MOVE), *a["lab"]), (max(0, 2 * f / MOVE - 1), *b["lab"])]))
    print("leg", k, "done", flush=True)

subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(tmp, "%04d.png"),
                "-vf", "split[a][b];[a]palettegen=max_colors=128:stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                "-loop", "0", OUT], check=True)
print(f"wrote {OUT}: {n} frames, {os.path.getsize(OUT) / 1e6:.1f} MB  (frames in {tmp})")
