"""docs/morph.gif: the same videos flying between the three maps (titles -> descriptions -> thumbnails -> titles),
coloured by their title-map cluster.  nice uv run python docs/morph.py
The description and thumbnail layouts are rotated/flipped/scaled onto the title layout first (Procrustes),
so the motion is videos changing neighbourhoods, not the whole picture spinning."""
import os, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.linalg import orthogonal_procrustes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIGHT = os.environ.get("BG") == "white"  # BG=white: ink on white (for nickita.org) -> docs/morph_white.gif
OUT = os.path.join(ROOT, "docs", "morph_white.gif" if LIGHT else "morph.gif")
PT = [("x", "<f4"), ("y", "<f4"), ("ch", "<u2"), ("m", "<u2"), ("cl", "<i2"), ("kind", "u1"), ("game", "u1")]
W, H, SS = 520, 432, 2                      # output size (site shows it ~280 px wide), supersampling
FPS, HOLD, MOVE = 15, 22, 24                 # frames: hold on each map, then fly to the next
SAMPLE = 110_000
NAMES = ["titles", "descriptions", "thumbnails"]


def load(sfx):
    d = os.path.join(ROOT, "web", "data" + sfx)
    r = np.fromfile(os.path.join(d, "points.bin"), dtype=PT)
    ids = open(os.path.join(d, "ids.txt")).read().split("\n")[:len(r)]
    return r, {v: i for i, v in enumerate(ids)}


def cluster_rgb(k):  # same as clusterRGB() in web/index.html
    h = (k * 0.61803) % 1
    c = np.clip(np.abs(((h[:, None] * 6 + np.array([0, 4, 2])) % 6) - 3) - 1, 0, 1)
    rgb = (0.5 + (c - 0.5) * 0.85)
    rgb[k < 0] = 90 / 255
    return rgb


t, ti = load("")
maps = [load(s) for s in ("_rich", "_thumb")]
keep = (t["game"] == 0)
common = [v for v, i in ti.items() if keep[i] and all(v in m[1] for m in maps)]
rng = np.random.default_rng(3)
common = [common[i] for i in rng.choice(len(common), min(SAMPLE, len(common)), replace=False)]
P = [np.c_[t["x"], t["y"]][[ti[v] for v in common]].astype(np.float64)]
for r, idx in maps:
    P.append(np.c_[r["x"], r["y"]][[idx[v] for v in common]].astype(np.float64))
# centre + unit size, then rotate/flip the other two onto the title layout
P = [(p - np.median(p, 0)) / np.sqrt(((p - np.median(p, 0)) ** 2).sum(1).mean()) for p in P]
for i in (1, 2):
    R, _ = orthogonal_procrustes(P[i], P[0])
    P[i] = P[i] @ R
col = cluster_rgb(t["cl"][[ti[v] for v in common]].astype(np.int64))
lit = t["cl"][[ti[v] for v in common]] >= 0
col[~lit] *= 0.75
if LIGHT:  # deeper colours read better on white; unclustered = light grey
    col = np.where(lit[:, None], col * 0.8, 0.78)
print(f"{len(common):,} videos on all three maps")

# frame: fit the 1-99% extent of all three layouts (a few far pieces may fly off the edge)
allp = np.concatenate(P)
lo, hi = np.quantile(allp, 0.01, 0), np.quantile(allp, 0.99, 0)
sc = min((W - 30) / (hi[0] - lo[0]), (H - 30) / (hi[1] - lo[1]))
off = np.array([(W - sc * (hi[0] - lo[0])) / 2, (H - sc * (hi[1] - lo[1])) / 2])
delay = rng.uniform(0, 0.35, len(common))  # each video leaves a little later than the last: a flock, not a jump
font = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 30 * SS, index=1)  # bold


def acc_ink(iy, ix, ok):  # videos per pixel (same 2x2 splat as the colours), as [H, W, 1]
    a = np.zeros((H * SS, W * SS))
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        np.add.at(a, (np.minimum(iy + dy, H * SS - 1), np.minimum(ix + dx, W * SS - 1)), 0.7)
    return a[..., None]


def render(xy, active):
    px = ((xy - lo) * sc + off) * SS
    ok = (px[:, 0] >= 0) & (px[:, 0] < W * SS) & (px[:, 1] >= 0) & (px[:, 1] < H * SS)
    ix, iy = px[ok, 0].astype(int), (H * SS - 1 - px[ok, 1]).astype(int)
    acc = np.zeros((H * SS, W * SS, 3))
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):  # 2x2 splat = one output pixel per video
        np.add.at(acc, (np.minimum(iy + dy, H * SS - 1), np.minimum(ix + dx, W * SS - 1)), col[ok] * 0.7)
    img = 1 - np.exp(-acc * 1.3)  # dense cores saturate instead of clipping to white
    if LIGHT:  # on white: each pixel goes from white toward the average colour of its videos, by how many there are
        cnt = acc_ink(iy, ix, ok)
        avg = acc / np.maximum(cnt, 1e-9)  # both carry the same 0.7 weight
        alpha = 1 - np.exp(-cnt * 1.2)
        img = 1 - alpha * (1 - avg)
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for i, n in enumerate(NAMES):  # the current map's name: the old one fades out, then the new one fades in
        a = max(0.0, 2 * active[i] - 1)
        if a > 0.02:
            g = int(240 * a) if not LIGHT else int(255 - 235 * a)
            d.text((22 * SS, (H - 22) * SS), "by " + n[:-1], font=font, fill=(g, g, g), anchor="ls")
    return im.resize((W, H), Image.BOX)  # box: Lanczos rings into dark halos on white


def ease(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


tmp = tempfile.mkdtemp()
n = 0
for k in range(3):
    a, b = P[k], P[(k + 1) % 3]
    act = np.eye(3)[k]
    for f in range(HOLD):
        render(a, act).save(os.path.join(tmp, f"{n:04d}.png")); n += 1
    for f in range(MOVE):
        u = f / (MOVE - 1)
        e = ease((u - delay) / 0.65)[:, None]
        mix = ease(np.array(u))
        render(a + (b - a) * e, act * (1 - mix) + np.eye(3)[(k + 1) % 3] * mix).save(os.path.join(tmp, f"{n:04d}.png")); n += 1
    print("map", NAMES[k], "done", flush=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(tmp, "%04d.png"),
                "-vf", "split[a][b];[a]palettegen=max_colors=128:stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                "-loop", "0", OUT], check=True)
print(f"wrote {OUT}: {n} frames, {os.path.getsize(OUT) / 1e6:.1f} MB  (frames in {tmp})")
