"""Run only map.py's layout (UMAP + HDBSCAN) on a big box and save data/map_state{SFX}.npz; then `map.py --apply` locally.
  MAP_SRC=rich MAP_FAST=1 python pipeline/box_layout.py"""
import time
import numpy as np
import map as M

t = time.time()
channels, rows, X = M.load()
key = np.array([r["cid"] + r["id"] for r in rows])
print(f"{len(rows):,} videos, X {X.shape}, load {time.time() - t:.0f}s", flush=True)
emb2, labels = M.layout(X, key, M.chan_codes(rows))
np.savez(M.STATE, emb2=emb2, labels=labels, key=key)
print(f"done {time.time() - t:.0f}s, {labels.max() + 1} clusters, {(labels < 0).mean():.0%} noise", flush=True)
