"""Try UMAP settings on a 100k sample of the title map and save a picture of each -> /tmp/lt_<name>.png
  nice uv run python experiments/layout_test.py"""
import os, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
import map as M

channels, rows, X = M.load()
st = np.load(M.STATE)
lab_full = st["labels"]
rng = np.random.default_rng(0)
idx = rng.choice(len(X), 100_000, replace=False)
Xs, lab = X[idx], lab_full[idx]
cols = rng.integers(70, 256, size=(lab_full.max() + 2, 3))
def pic(E, name):
    lo, hi = np.quantile(E, .005, 0), np.quantile(E, .995, 0)
    S = 900
    g = ((E - lo) / (hi - lo).max() * (S - 1)).astype(int)
    ok = (g >= 0).all(1) & (g < S).all(1)
    img = np.zeros((S, S, 3)); cnt = np.zeros((S, S))
    c = np.where(lab[:, None] >= 0, cols[lab], 60)
    np.add.at(img, (g[ok, 1], g[ok, 0]), c[ok]); np.add.at(cnt, (g[ok, 1], g[ok, 0]), 1)
    img = img / np.maximum(cnt, 1)[..., None] * np.minimum(1, cnt / 2)[..., None] ** .5
    Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(f"/tmp/lt_{name}.png")
pic(st["emb2"][idx], "current_full")  # the live map's positions, same sample
import umap
V = {"A_current": dict(n_neighbors=30, min_dist=0.05),
     "B_pca_init": dict(n_neighbors=30, min_dist=0.05, init="pca"),
     "C_more_repulsion": dict(n_neighbors=15, min_dist=0.1, negative_sample_rate=15, init="pca"),
     "D_local": dict(n_neighbors=10, min_dist=0.0, negative_sample_rate=10, repulsion_strength=1.5, init="pca")}
from sklearn.neighbors import NearestNeighbors
true = NearestNeighbors(n_neighbors=13, metric="cosine").fit(Xs[:20000]).kneighbors(Xs[:2000], return_distance=False)[:, 1:]
for k, p in V.items():
    t = time.time()
    E = umap.UMAP(metric="cosine", random_state=None, low_memory=True, verbose=False, **p).fit_transform(Xs)
    m = NearestNeighbors(n_neighbors=13).fit(E[:20000]).kneighbors(E[:2000], return_distance=False)[:, 1:]
    keep = np.mean([len(set(a) & set(b)) / 12 for a, b in zip(true, m)])
    # how much of the picture is empty vs packed: share of occupied grid cells holding the busiest 50% of points
    pic(E, k)
    print(f"{k}: {time.time() - t:.0f}s  12-NN kept {keep:.2f}", flush=True)
