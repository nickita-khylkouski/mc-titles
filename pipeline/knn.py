"""The neighbour graph behind the map, and the clean-up that uses it (imported by map.py).

  graph(X, key)        every video's 48 true nearest neighbours (pynndescent, cosine), cached in data/knn{SFX}.npz
  cross(idx, sim, ch)  a neighbour list built mostly from OTHER channels' videos. Plain nearest neighbours of a title
                       are mostly the same channel's other titles ("Hermitcraft 7: Episode 41" -> "Episode 40"), so
                       UMAP drew thousands of one-channel islands and dust. Asking for similar videos from other
                       channels lays the map out by topic: half the islands, a third fewer strays, and true
                       neighbours land as close as before (layout test 2026-10-24: near1 0.825 -> 0.832)
  tidy(xy, idx)        a speck (island under 100 videos) moves as one piece next to its videos' most similar main-map
                       videos; a video alone in empty space moves beside its own most similar one
  vote8(emb8, labels)  after HDBSCAN: an unclustered video joins the cluster its nearest clustered videos agree on
  vote(cl, nb)         after single-channel series are folded out: the same, using other channels' similar videos
                       (2026-10-24 title map: unclustered 54% -> 36%, newly joined videos sit about as close to their
                       cluster's centre as its core members, cos 0.67 vs 0.71)
"""
import hashlib, os, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("MAP_SRC", "title")
SFX = "" if SRC == "title" else "_" + SRC
D = os.path.join(ROOT, "data")
K = 48
PATH = os.path.join(D, f"knn{SFX}.npz")


def graph(X, key):
    """(idx int32 [N,K], sim float32 [N,K]) without self; X is L2-normalised, key = the rows' ids (array of str)."""
    h = hashlib.sha1("\n".join(map(str, key)).encode()).hexdigest()
    if os.path.exists(PATH):
        z = np.load(PATH)
        if str(z["key"]) == h and z["idx"].shape[1] == K:
            return z["idx"], z["sim"].astype(np.float32)
    import pynndescent
    t = time.time()
    nn = pynndescent.NNDescent(X, metric="cosine", n_neighbors=K + 1, low_memory=True, random_state=0, n_jobs=-1)
    idx, dist = nn.neighbor_graph
    notself = idx != np.arange(len(X))[:, None]  # exact duplicates can come before self
    order = np.argsort(~notself, axis=1, kind="stable")[:, :K]
    idx = np.take_along_axis(idx, order, 1).astype(np.int32)
    sim = (1 - np.take_along_axis(dist, order, 1)).astype(np.float16)
    np.savez(PATH, idx=idx, sim=sim, key=h)
    print(f"knn graph {len(X):,} x {K} in {time.time() - t:.0f}s", flush=True)
    return idx, sim.astype(np.float32)


def cross(idx, sim, ch, other, same):
    """UMAP precomputed_knn: self + the `other` closest from other channels + the `same` closest from its own channel
    (topped up from whichever side has more, so every row is full). Returns (indices int64, distances float32)."""
    n = len(idx)
    s = ch[idx] == ch[:, None]
    ro, rs = np.cumsum(~s, 1) - 1, np.cumsum(s, 1) - 1
    grp = np.where((~s & (ro < other)) | (s & (rs < same)), 0, 1)
    order = np.argsort(grp * 1000 + np.arange(idx.shape[1])[None, :], axis=1, kind="stable")[:, :other + same]
    order.sort(1)  # back to closest-first
    ii = np.c_[np.arange(n), np.take_along_axis(idx, order, 1)].astype(np.int64)
    dd = np.c_[np.zeros(n), np.clip(1 - np.take_along_axis(sim, order, 1), 0, 2)].astype(np.float32)
    return ii, dd


def _loose(xy, min_island, ch=None, solo=5000, grid=250):
    """(specks, strays, md). A speck is a piece of the map under min_island videos, with pieces cut where there is a
    gap you can see at the fit-to-screen zoom (the map binned into grid x grid cells, touching cells = one piece).
    A piece under `solo` videos that is mostly (> 60%) one channel is a speck too: that's one channel's series, not a topic.
    A stray is a video in a big piece that sits alone (10th-closest video over 4x the typical distance md)."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    d10 = cKDTree(xy).query(xy, k=11)[0][:, 10]
    md = np.median(d10)
    lo, ext = xy.min(0), np.ptp(xy, 0).max()
    ij = ((xy - lo) / ext * (grid - 1)).astype(int)
    occ = np.zeros((grid, grid), bool)
    occ[ij[:, 0], ij[:, 1]] = True
    lab = ndimage.label(occ, structure=np.ones((3, 3)))[0][ij[:, 0], ij[:, 1]]
    size = np.bincount(lab)
    small = size < min_island
    if ch is not None:
        pair = lab.astype(np.int64) * (int(ch.max()) + 1) + ch
        u, c = np.unique(pair, return_counts=True)
        top = np.zeros(len(size)); np.maximum.at(top, u // (int(ch.max()) + 1), c)
        small |= (size < solo) & (top > 0.6 * size)
    speck = small[lab]
    return np.where(speck, lab, -1), ~speck & (d10 > 4 * md), md


def tidy(xy, idx, X=None, ch=None, min_island=1000, rounds=4, seed=1):
    """idx may hold -1 (neighbour not on the map). A speck moves as one piece (so a channel's series stays together)
    to where most of its videos' closest main-map videos are; a stray moves right beside its own closest one. A speck
    whose videos only have each other as neighbours (near-identical titles) goes beside the main-map video closest to
    its mean vector (X = the rows' unit vectors; ch = channel codes, for one-channel pieces). Moving videos one by one scattered each speck and cost near1
    0.834 -> 0.810; as pieces 0.827 (test 2026-10-24). Returns new xy and which videos moved."""
    xy = xy.astype(np.float64).copy()
    g, moved = np.random.default_rng(seed), np.zeros(len(xy), bool)
    for _ in range(rounds):
        speck, stray, md = _loose(xy, min_island, ch)
        loose = (speck >= 0) | stray
        ok = (idx >= 0) & ~loose[np.maximum(idx, 0)]
        has = ok.any(1)
        tgt = idx[np.arange(len(xy)), np.argmax(ok, 1)]
        go = stray & has
        xy[go] = xy[tgt[go]] + g.normal(0, 0.3 * md, (go.sum(), 2))
        moved |= go
        sm = np.flatnonzero(speck >= 0)
        sm = sm[np.argsort(speck[sm], kind="stable")]
        main = np.flatnonzero(~loose)
        for grp in np.split(sm, np.flatnonzero(np.diff(speck[sm])) + 1):
            T = xy[tgt[grp[has[grp]]]]
            if len(T):
                a = T[np.argmin(np.sqrt(((T[:, None] - T[None]) ** 2).sum(-1)).sum(1))]  # medoid of the targets
            elif X is not None:
                a = xy[main[np.argmax(X[main] @ X[grp].mean(0))]]
            else:
                continue
            xy[grp] += a - xy[grp].mean(0)
            moved[grp] = True
    speck, stray, _ = _loose(xy, min_island, ch)
    print(f"tidy: moved {moved.sum():,} videos ({moved.mean():.1%}); still loose {((speck >= 0) | stray).mean():.2%}", flush=True)
    return xy.astype(np.float32), moved


def vote8(emb8, labels, k=15, share=0.5):
    """HDBSCAN only labels the dense cores (~40% of videos left out). An unclustered video whose k nearest clustered
    videos in the 8-d clustering space mostly (>= share) agree joins that cluster, unless it sits far from all of them."""
    from scipy.spatial import cKDTree
    labels = labels.copy()
    L = np.flatnonzero(labels >= 0)
    if not len(L):
        return labels
    tree = cKDTree(emb8[L])
    cap = 3 * np.quantile(tree.query(emb8[L], k=2)[0][:, 1], 0.99)
    noise = np.flatnonzero(labels < 0)
    d, i = tree.query(emb8[noise], k=k)
    nl = labels[L[i]]
    nc = int(labels.max()) + 1
    tally = np.zeros((len(noise), nc), np.int32)
    np.add.at(tally, (np.repeat(np.arange(len(noise)), k), nl.ravel()), 1)
    best = tally.argmax(1)
    win = (tally[np.arange(len(noise)), best] >= share * k) & (d[:, 0] <= cap)
    labels[noise[win]] = best[win]
    return labels


def vote(cl, nb, need=8, min_votes=5, rounds=4):
    """cl int array (-1 = unclustered), nb [N,k] neighbour ids (-1 = none; map.py passes other-channel neighbours, so a
    channel's own series joins the topic other channels' similar videos are in). Joins when >= need neighbours agree."""
    cl = cl.astype(np.int32).copy()
    nc = int(cl.max()) + 1
    for _ in range(rounds):
        lab = np.where(nb >= 0, cl[np.maximum(nb, 0)], -1)
        todo = np.flatnonzero((cl < 0) & ((lab >= 0).sum(1) >= min_votes))
        if not len(todo):
            break
        L = lab[todo]
        m = L >= 0
        tally = np.zeros((len(todo), nc), np.int32)
        np.add.at(tally, (np.repeat(np.arange(len(todo)), m.sum(1)), L[m]), 1)
        best = tally.argmax(1)
        win = tally[np.arange(len(todo)), best] >= need
        if not win.any():
            break
        cl[todo[win]] = best[win]
    return cl
