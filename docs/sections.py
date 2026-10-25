"""docs/sections.gif: the title map with every topic in its own colour, then one named topic lit up at a time.
White background (for nickita.org; BG=black for the README look).  nice uv run python docs/sections.py"""
import json, os, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DARK = os.environ.get("BG") == "black"
OUT = os.path.join(ROOT, "docs", "sections_black.gif" if DARK else "sections.gif")
PT = [("x", "<f4"), ("y", "<f4"), ("ch", "<u2"), ("m", "<u2"), ("cl", "<i2"), ("kind", "u1"), ("game", "u1")]
W, H, SS = 400, 440, 2  # about the map's own shape, so it fills the frame
FPS, INTRO, FADE, HOLD = 15, 30, 6, 17
TOUR = ["Hermitcraft episodes", "Bed Wars", "I Survived 100 Days as…", "Pixelmon", "Updates & snapshots", "House tutorials",
        "Lucky Block mod battles", "Herobrine animations", "FNAF roleplay", "Hunger Games"]  # big and tight enough to see at 280 px
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
big, small = ImageFont.truetype(FONT, 30 * SS, index=1), ImageFont.truetype(FONT, 19 * SS)

d = os.path.join(ROOT, "web", "data")
r = np.fromfile(os.path.join(d, "points.bin"), dtype=PT)
meta = json.load(open(os.path.join(d, "meta.json")))
r = r[r["game"] == 0]  # Minecraft videos only
byname = {c["name"]: c for c in meta["clusters"]}
nshown = sum(1 for c in meta["clusters"] if c.get("type") != "game")
xy = np.c_[r["x"], r["y"]].astype(np.float64)
lo, hi = np.quantile(xy, [0.03, 0.005], 0).diagonal(), np.quantile(xy, 0.985, 0)  # the main body (plus the Hermitcraft island below it); a few far pieces fall off the edge
pad = np.array([8, 86])  # room at the bottom for the label
sc = min((W - 2 * pad[0]) / (hi[0] - lo[0]), (H - pad[1] - 16) / (hi[1] - lo[1]))
off = np.array([(W - sc * (hi[0] - lo[0])) / 2, pad[1] + (H - pad[1] - 16 - sc * (hi[1] - lo[1])) / 2])
px = ((xy - lo) * sc + off) * SS
ok = (px[:, 0] >= 0) & (px[:, 0] < W * SS - 1) & (px[:, 1] >= 0) & (px[:, 1] < H * SS - 1)
ix, iy = px[:, 0].astype(int), (H * SS - 1 - px[:, 1]).astype(int)


def cluster_rgb(k):  # same as clusterRGB() in web/index.html
    h = (k * 0.61803) % 1
    c = np.clip(np.abs(((h[:, None] * 6 + np.array([0, 4, 2])) % 6) - 3) - 1, 0, 1)
    return 0.5 + (c - 0.5) * 0.85


cl = r["cl"].astype(np.int64)
col = cluster_rgb(cl) * (1.0 if DARK else 0.82)
BGC = 0.0 if DARK else 1.0
GREY = 0.30 if DARK else 0.86


def layer(mask, rgb, k, sp=2):
    """(average colour, coverage) of the videos in mask, splatted sp x sp at SS x size."""
    m = mask & ok & (ix < W * SS - sp) & (iy < H * SS - sp)
    acc, cnt = np.zeros((H * SS, W * SS, 3)), np.zeros((H * SS, W * SS, 1))
    for dx, dy in [(a, b) for a in range(sp) for b in range(sp)]:
        np.add.at(acc, (iy[m] + dy, ix[m] + dx), rgb[m])
        np.add.at(cnt, (iy[m] + dy, ix[m] + dx, 0), 1)
    return acc / np.maximum(cnt, 1e-9), 1 - np.exp(-cnt * k)


def over(img, lay):
    c, a = lay
    return img * (1 - a) + c * a


def finish(img, title, sub):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    ink, dim = ((240,) * 3, (150,) * 3) if DARK else ((20,) * 3, (110,) * 3)
    dr.text((22 * SS, (H - 44) * SS), title, font=big, fill=ink, anchor="ls")
    dr.text((22 * SS, (H - 18) * SS), sub, font=small, fill=dim, anchor="ls")
    return np.asarray(im.resize((W, H), Image.BOX)).astype(np.float32)


blank = np.full((H * SS, W * SS, 3), BGC)
greyall = layer(np.ones(len(r), bool), np.full((len(r), 3), GREY), 0.9)
full = over(over(blank, layer(cl < 0, np.full((len(r), 3), GREY), 0.9)), layer(cl >= 0, col, 0.9))
frames = [finish(full, f"{nshown} topics", f"{len(r):,} Minecraft videos, mapped by title")]
base = over(blank, greyall)
for name in TOUR:
    c = byname[name]
    m = cl == c["id"]
    lit = layer(m, col, 2.0, sp=3)  # a bit fatter than the grey dots
    glow = gaussian_filter(lit[1][..., 0], 7 * SS)[..., None]  # soft halo in the topic's colour, so it reads at a glance
    tint = np.median(col[m], 0)
    img = over(base, (np.broadcast_to(tint, base.shape), np.clip(glow * 2.2, 0, 0.45)))
    frames.append(finish(over(img, lit), name, f"{c['size']:,} videos from {c['channels']} channels"))
    print(name, flush=True)

tmp = tempfile.mkdtemp()
n = 0
def save(a):
    global n
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(os.path.join(tmp, f"{n:04d}.png")); n += 1
seq = frames + [frames[0]]
for i in range(len(seq) - 1):
    for _ in range(INTRO if i == 0 else HOLD):
        save(seq[i])
    for f in range(1, FADE + 1):
        u = f / (FADE + 1); u = u * u * (3 - 2 * u)
        save(seq[i] * (1 - u) + seq[i + 1] * u)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(tmp, "%04d.png"),
                "-vf", "split[a][b];[a]palettegen=max_colors=128:stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                "-loop", "0", OUT], check=True)
print(f"wrote {OUT}: {n} frames, {os.path.getsize(OUT) / 1e6:.1f} MB  (frames in {tmp})")
