"""How much of the map is one channel talking to itself: for every video, the share of its 30 nearest map
neighbours (2-d layout) that come from the same channel. 0.03 would be "mixed in", 1.0 is a private island.
  MAP_SRC=thumb python3 pipeline/islands.py"""
import json, os
import numpy as np
from scipy.spatial import cKDTree

SRC = os.environ.get("MAP_SRC", "title")
W = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "data" + ("" if SRC == "title" else "_" + SRC))
meta = json.load(open(os.path.join(W, "meta.json")))
r = np.fromfile(os.path.join(W, "points.bin"), dtype=[("x", "<f4"), ("y", "<f4"), ("ch", "<u2"), ("m", "<u2"),
                                                      ("cl", "<i2"), ("kind", "u1"), ("game", "u1")])
xy = np.c_[r["x"], r["y"]]
_, nb = cKDTree(xy).query(xy, k=31)
same = (r["ch"][nb[:, 1:]] == r["ch"][:, None]).mean(1)
print(f"{SRC}: {len(r):,} videos; median same-channel neighbours {np.median(same):.0%}; "
      f"{(same > .5).mean():.0%} of videos sit where most neighbours are their own channel")
names = [c["name"] for c in meta["channels"]]
cnt = np.bincount(r["ch"], minlength=len(names))
isl = np.bincount(r["ch"], weights=same, minlength=len(names)) / np.maximum(cnt, 1)
order = [i for i in np.argsort(-isl * np.sqrt(cnt)) if cnt[i] >= 200][:15]
print("biggest private islands (channel: videos, avg same-channel neighbours):")
for i in order:
    print(f"   {names[i]}: {cnt[i]}, {isl[i]:.0%}")
