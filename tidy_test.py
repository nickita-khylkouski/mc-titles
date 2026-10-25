"""Offline test of tidy variants on the saved title layout: near1 (share of 12 true neighbours in the 2-d closest 1%), loose share."""
import sys, time
import numpy as np
from scipy.spatial import cKDTree
import knn as K
z = np.load("data/map_state.npz", allow_pickle=True)
xy0 = z["emb2"].astype(np.float32)
idx = np.load("data/knn.npz")["idx"]
N = len(xy0)
smp = np.random.default_rng(0).choice(N, 3000, replace=False)

def near1(xy, sub=None):
    s = smp if sub is None else sub
    tree = cKDTree(xy)
    r1 = tree.query(xy[s], k=N // 100)[0][:, -1]
    d = np.sqrt(((xy[idx[s, :12]] - xy[s][:, None]) ** 2).sum(-1))
    return (d <= r1[:, None]).mean(1)

def group_tidy(xy, idx, min_island=100, rounds=3, jit=0.3, seed=1):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    xy = xy.astype(np.float64).copy(); g = np.random.default_rng(seed); moved = np.zeros(len(xy), bool)
    for _ in range(rounds):
        tree = cKDTree(xy); d10 = tree.query(xy, k=11)[0][:, 10]; md = np.median(d10)
        p = tree.query_pairs(4 * md, output_type="ndarray")
        _, lab = connected_components(coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(len(xy),) * 2), directed=False)
        size = np.bincount(lab)[lab]
        speck, stray = size < min_island, (size >= min_island) & (d10 > 4 * md)
        loose = speck | stray
        ok = (idx >= 0) & ~loose[np.maximum(idx, 0)]
        has = ok.any(1)
        tgt = idx[np.arange(len(xy)), np.argmax(ok, 1)]
        # strays: right beside their closest on-map video
        go = stray & has
        xy[go] = xy[tgt[go]] + g.normal(0, jit * md, (go.sum(), 2)); moved |= go
        # specks: move as one piece to where most members' closest on-map videos are
        sm = np.flatnonzero(speck & has)
        order = np.argsort(lab[sm], kind="stable"); sm = sm[order]
        cuts = np.flatnonzero(np.diff(lab[sm])) + 1
        for grp in np.split(sm, cuts):
            T = xy[tgt[grp]]
            if len(grp) > 1:
                dd = np.sqrt(((T[:, None] - T[None]) ** 2).sum(-1)).sum(1)
                a = T[np.argmin(dd)]
            else:
                a = T[0]
            members = np.flatnonzero(lab == lab[grp[0]])
            c = xy[members].mean(0)
            xy[members] = a + (xy[members] - c) + g.normal(0, jit * md, (len(members), 2)) * 0
            moved[members] = True
    tree = cKDTree(xy); d10 = tree.query(xy, k=11)[0][:, 10]; md = np.median(d10)
    p = tree.query_pairs(4 * md, output_type="ndarray")
    _, lab = connected_components(coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(len(xy),) * 2), directed=False)
    loose = (np.bincount(lab)[lab] < min_island) | (d10 > 4 * md)
    print(f"  moved {moved.mean():.1%} still loose {loose.mean():.2%}")
    return xy.astype(np.float32), moved

t = time.time()
sp, st, _ = K._loose(xy0, 1000); loose0 = (sp >= 0) | st
b = near1(xy0); print(f"raw  near1 {b.mean():.4f}  loose-sample {b[loose0[smp]].mean():.4f}  ({loose0.mean():.2%} loose)", flush=True)
for mi in (1000, 2000):
    xy, mv = K.tidy(xy0, idx, min_island=mi)
    v = near1(xy); print(f"min_island {mi}: near1 {v.mean():.4f}  loose-sample {v[loose0[smp]].mean():.4f}  {time.time()-t:.0f}s", flush=True)
    np.save(f"/tmp/tidy{mi}.npy", xy)
