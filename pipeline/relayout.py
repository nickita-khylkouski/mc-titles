"""Redo only the 2-d picture of a map with the less-squashed UMAP settings, keeping its clusters (labels) as they are.
Old state is copied to data/map_state{SFX}.pre-relayout.npz first. Then run `map.py --apply` for that map.
  MAP_SRC=title nice uv run python pipeline/relayout.py"""
import os, shutil, time
import numpy as np
import map as M

t = time.time()
channels, rows, X = M.load()
key = np.array([r["cid"] + r["id"] for r in rows])
st = np.load(M.STATE)
assert np.array_equal(st["key"], key), "videos changed since the last layout"
bak = M.STATE.replace(".npz", ".pre-relayout.npz")
if not os.path.exists(bak):
    shutil.copy(M.STATE, bak)
print(f"{len(rows):,} videos, load {time.time() - t:.0f}s", flush=True)
emb2 = M.layout2(X, M.K.graph(X, key), M.chan_codes(rows))
np.savez(M.STATE, emb2=emb2, labels=st["labels"], key=key)
print(f"done {time.time() - t:.0f}s", flush=True)
