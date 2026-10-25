"""Step 4: 2D map + clusters + names + channel paths, written to web/data/ for the UI.

  python map.py           full run: UMAP + HDBSCAN (slow), cached in data/map_state.npz, then --apply
  python map.py --apply   reuse the cached layout; just re-apply names/drops and rewrite web/data (seconds)

Layout: UMAP (cosine) 512-d -> 2-d for the picture (layout2), and -> 8-d for HDBSCAN (clustering in the 2-d
picture mostly finds layout artefacts). Videos before May 2009 (pre-Minecraft uploads) are left out.

Clusters:
  * a cluster where one channel makes > SERIES_SHARE of the videos is that channel's own series, not a
    community topic -> folded into unclustered (dots stay, no label)
  * names + drops live in data/cluster_names.json, one entry per cluster with its 512-d centroid.
    Hand-written names carry over to a rebuilt map by matching centroids (cosine >= MATCH); clusters with
    no match get a placeholder name from gpt-5.4-mini and "auto": true, to be reviewed
  * "drop": true removes that cluster's videos from the map entirely (off-topic: vlogs, music, IRL...)
  * "type": "game" marks a cluster about another game (Roblox, Fortnite...); the UI hides those by default
  data/clusters_review.txt lists every cluster with top channels + sample titles, for naming by hand.

Outputs (web/data/):
  points.bin   per video: float32 x, float32 y, uint16 channel, uint16 month (months since 2005-01),
               int16 cluster (-1 = unclustered), uint8 kind (0 long, 1 short), uint8 game (1 = other-game cluster)
               -> 16 bytes
  titles.txt   one title per line, same order as points.bin
  ids.txt      YouTube video id per line, same order
  meta.json    channels, clusters, paths, bounds
"""
import glob, json, os, sys, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "data")
# MAP_SRC=thumb: same pipeline on Gemini thumbnail embeddings (data/emb_thumb), written to web/data_thumb
SRC = os.environ.get("MAP_SRC", "title")
SFX = "" if SRC == "title" else "_" + SRC
EMB = os.path.join(D, "emb" + SFX)
OUT = os.path.join(ROOT, "web", "data" + SFX)
STATE = os.path.join(D, f"map_state{SFX}.npz")
NAMES = os.path.join(D, f"cluster_names{SFX}.json")
REVIEW = os.path.join(D, f"clusters_review{SFX}.txt")
sys.path.insert(0, ROOT)
from classify import env  # noqa: E402
import knn as K  # noqa: E402

SAMPLE = int(os.environ.get("MAP_SAMPLE", 0))  # >0: random subset for quick tests
FIRST_MONTH = (2009 - 2005) * 12 + 4           # May 2009, Minecraft's first public release
SERIES_SHARE = 0.6
MIN_SAMPLES = int(os.environ.get("MAP_MIN_SAMPLES", 10))
MATCH = 0.9


def month_of(date):
    return (int(date[:4]) - 2005) * 12 + int(date[5:7]) - 1


def load():
    channels = {c["id"]: c for c in json.load(open(os.path.join(D, "channels.json")))}
    rows, vecs = [], []
    for p in sorted(glob.glob(os.path.join(EMB, "*.npy"))):
        cid = os.path.basename(p)[:-4]
        if cid not in channels:
            continue
        e = np.load(p)
        vids = [json.loads(l) for l in open(os.path.join(D, "videos", cid + ".jsonl"))]
        assert len(vids) == len(e), cid
        ok = np.abs(e).sum(1) > 0  # zero row = no thumbnail
        mp = os.path.join(D, "is_mc", cid + ".json")  # tag_mc.py; missing = assume Minecraft
        pmc = json.load(open(mp)) if os.path.exists(mp) else [1.0] * len(vids)
        for v, q in zip(vids, pmc):
            v["mc"] = q
            v.pop("desc", None)  # unused here; full descriptions for 500k videos cost GBs of RAM
        keep = [i for i, v in enumerate(vids) if month_of(v["date"]) >= FIRST_MONTH and ok[i]]
        rows += [vids[i] for i in keep]
        vecs.append(e[keep])
    X = np.concatenate(vecs).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    if SAMPLE and SAMPLE < len(rows):
        idx = np.random.default_rng(0).choice(len(rows), SAMPLE, replace=False)
        rows, X = [rows[i] for i in idx], X[idx]
    return channels, rows, X


def chan_codes(rows):
    return np.unique([r["cid"] for r in rows], return_inverse=True)[1]


def layout2(X, nb, ch, seed=None):
    """The 2-d picture. n_neighbors=30 squashed 500k videos into one ball (layout_test.py); 15 neighbours with more
    repulsion and a PCA start spread it out, and taking 12 of them from other channels (knn.cross) lays it out by
    topic, not by each channel's title template."""
    import umap
    ii, dd = K.cross(*nb, ch, other=12, same=2)
    return umap.UMAP(n_neighbors=15, min_dist=0.0, negative_sample_rate=10, repulsion_strength=1.5, init="pca", metric="cosine",
                     precomputed_knn=(ii, dd, None), random_state=seed, low_memory=True).fit_transform(X).astype(np.float32)


def layout(X, key, ch):
    import hdbscan, umap
    seed = None if os.environ.get("MAP_FAST") else 42  # MAP_FAST=1: unseeded, so UMAP uses every core (box runs)
    nb = K.graph(X, key)
    emb2 = layout2(X, nb, ch, seed)
    ii, dd = K.cross(*nb, ch, other=26, same=3)
    emb8 = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=8, metric="cosine", precomputed_knn=(ii, dd, None),
                     random_state=seed, low_memory=True).fit_transform(X)
    labels = hdbscan.HDBSCAN(min_cluster_size=max(30, len(X) // 1500), min_samples=MIN_SAMPLES, core_dist_n_jobs=6).fit_predict(emb8)
    raw = (labels < 0).mean()
    labels = K.vote8(emb8, labels.astype(np.int32))
    print(f"{labels.max() + 1} clusters; unclustered {raw:.1%} -> {(labels < 0).mean():.1%} after the 8-d vote", flush=True)
    return emb2, labels.astype(np.int32)


def gpt_name(titles):
    body = json.dumps({"model": "gpt-5.4-mini", "response_format": {"type": "json_object"}, "messages": [
        {"role": "system", "content": "These YouTube titles are one cluster of a map of Minecraft YouTube. "
         "Name the shared theme/format in 2-5 words, the way a fan would say it. Reply JSON: {\"name\": \"...\"}"},
        {"role": "user", "content": "\n".join(titles)}]}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body, headers={
        "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(json.load(r)["choices"][0]["message"]["content"])["name"]
    except Exception as e:  # placeholder is fine; it gets reviewed anyway
        print("name failed:", e, file=sys.stderr)
        return "unnamed"


def kind_of(titles):
    """First-pass tag for a new cluster (reviewed by hand later): mc / game / irl (irl gets dropped)."""
    from decide import choice, decide
    try:
        return decide("Video titles from one cluster of a map of Minecraft YouTube:\n" + "\n".join(titles), [
            choice("kind", "What are most of these videos about?", {
                "mc": "Minecraft (gameplay, mods, servers, animations, memes, Shorts)",
                "game": "a different video game (Roblox, Fortnite, GTA, Terraria...)",
                "irl": "real life: vlogs, food, music, unboxings, challenges with no game"})])["kind"][0]
    except Exception as e:
        print("kind failed:", e, file=sys.stderr)
        return "mc"


def resolve_clusters(rows, X, labels, channels):
    """Fold single-channel series into noise, then attach names/drops (carried over by centroid)."""
    labels = labels.copy()
    series = []
    for k in sorted(set(labels) - {-1}):
        idx = np.flatnonzero(labels == k)
        cid, n = Counter(rows[i]["cid"] for i in idx).most_common(1)[0]
        if n / len(idx) > SERIES_SHARE:
            series.append((channels[cid]["name"], len(idx)))
            labels[idx] = -1
    print(f"folded {len(series)} single-channel series into unclustered", flush=True)

    ids = sorted(set(labels) - {-1})
    cent = np.stack([X[labels == k].mean(0) for k in ids]) if ids else np.zeros((0, X.shape[1]))
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-8
    old = json.load(open(NAMES)) if os.path.exists(NAMES) else []
    info = {}
    if old:
        oc = np.array([o["centroid"] for o in old], dtype=np.float32)
        oc /= np.linalg.norm(oc, axis=1, keepdims=True) + 1e-8
        sim = cent @ oc.T
        taken = set()
        for flat in np.argsort(-sim, axis=None):
            i, j = divmod(int(flat), len(old))
            if sim[i, j] < MATCH:
                break
            if ids[i] in info or j in taken:
                continue
            info[ids[i]] = {k: old[j][k] for k in ("name", "drop", "auto", "type") if k in old[j]}
            taken.add(j)
    todo = [k for k in ids if k not in info]
    print(f"{len(ids)} clusters: {len(info)} named from before, {len(todo)} new", flush=True)

    def sample(k, n):
        idx = np.flatnonzero(labels == k)
        return [rows[i]["title"] for i in np.random.default_rng(int(k)).choice(idx, min(n, len(idx)), replace=False)]

    if todo:
        env()
        with ThreadPoolExecutor(6) as ex:
            for k, name, kind in zip(todo, ex.map(lambda k: gpt_name(sample(k, 40)), todo),
                                     ex.map(lambda k: kind_of(sample(k, 40)), todo)):
                info[k] = {"name": name, "drop": kind == "irl", "auto": True, "type": "game" if kind == "game" else "mc"}

    json.dump([{"id": int(k), **info[k], "centroid": [round(float(v), 4) for v in cent[i]]}
               for i, k in enumerate(ids)], open(NAMES, "w"), ensure_ascii=False)

    with open(REVIEW, "w") as f:
        for k in ids:
            idx = np.flatnonzero(labels == k)
            top = Counter(channels[rows[i]["cid"]]["name"] for i in idx).most_common(4)
            yrs = sorted(int(rows[i]["date"][:4]) for i in idx)
            f.write(f"## {k} | {info[k]['name']}{' (auto)' if info[k].get('auto') else ''}"
                    f"{' DROP' if info[k].get('drop') else ''} | {len(idx)} videos, "
                    f"{len({rows[i]['cid'] for i in idx})} channels, {yrs[len(yrs)//10]}-{yrs[-len(yrs)//10-1]} | "
                    + ", ".join(f"{n} {c}" for c, n in top) + "\n")
            f.write("   " + " / ".join(sample(k, 14)) + "\n")
    rng = np.random.default_rng(0)  # sample video ids per cluster, for sheets.py contact sheets
    json.dump({int(k): [rows[i]["id"] for i in rng.choice(np.flatnonzero(labels == k), min(24, int((labels == k).sum())), replace=False)]
               for k in ids}, open(REVIEW.replace(".txt", "_ids.json"), "w"))
    return labels, ids, info


def write(channels, rows, X, emb2, labels):
    labels, ids, info = resolve_clusters(rows, X, labels, channels)
    dropped = {k for k in ids if info[k].get("drop")}
    keep = np.flatnonzero(~np.isin(labels, list(dropped))) if dropped else np.arange(len(rows))
    print(f"dropping {len(rows) - len(keep):,} videos in {len(dropped)} off-topic clusters", flush=True)
    # the neighbour graph, re-indexed to the kept videos (-1 = neighbour was dropped)
    idx, sim = K.graph(X, np.array([r["cid"] + r["id"] for r in rows]))
    other = K.cross(idx, sim, chan_codes(rows), other=16, same=0)[0][:, 1:]
    inv = np.full(len(rows), -1, np.int64)
    inv[keep] = np.arange(len(keep))
    idx, other = inv[idx[keep]], inv[other[keep]]
    rows = [rows[i] for i in keep]
    emb2, labels = emb2[keep], labels[keep]
    emb2, _ = K.tidy(emb2, idx, X[keep], chan_codes(rows))  # small islands and lone dots move beside their closest videos in the main map
    # full vectors in points.bin order, for serve.py (search by meaning, true nearest neighbours); not served raw
    np.save(os.path.join(D, f"vecs{SFX}.npy"), X[keep].astype(np.float16))

    # renumber clusters 0..n-1 so the UI can index them
    live = [k for k in ids if k not in dropped]
    remap = {k: i for i, k in enumerate(live)}
    labels = np.array([remap.get(int(l), -1) for l in labels], dtype=np.int16)
    before = (labels < 0).mean()
    labels = K.vote(labels, other).astype(np.int16)  # unclustered videos join the cluster their neighbours are in
    print(f"unclustered {before:.1%} -> {(labels < 0).mean():.1%} after the neighbour vote", flush=True)
    clusters = []
    for k in live:
        idx = np.flatnonzero(labels == remap[k])
        cx, cy = np.median(emb2[idx], axis=0)
        clusters.append({"id": remap[k], "name": info[k]["name"], "auto": bool(info[k].get("auto")),
                         "type": info[k].get("type", "mc"),
                         "size": int(len(idx)), "channels": len({rows[i]["cid"] for i in idx}),
                         "x": float(cx), "y": float(cy)})

    cids = sorted({r["cid"] for r in rows}, key=lambda c: -channels[c]["subs"])
    cidx = {c: i for i, c in enumerate(cids)}
    rec = np.zeros(len(rows), dtype=[("x", "<f4"), ("y", "<f4"), ("ch", "<u2"), ("m", "<u2"),
                                     ("cl", "<i2"), ("kind", "u1"), ("pad", "u1")])
    rec["x"], rec["y"], rec["cl"] = emb2[:, 0], emb2[:, 1], labels
    rec["ch"] = [cidx[r["cid"]] for r in rows]
    rec["m"] = [month_of(r["date"]) for r in rows]
    rec["kind"] = [r["kind"] == "short" for r in rows]
    game = np.array([c["type"] == "game" for c in clusters] + [False])  # [-1] -> unclustered
    rec["pad"] = game[labels] | (np.array([r.get("mc", 1.0) for r in rows]) < 0.3)  # other-game cluster, or the video itself isn't Minecraft
    os.makedirs(OUT, exist_ok=True)
    rec.tofile(os.path.join(OUT, "points.bin"))
    with open(os.path.join(OUT, "titles.txt"), "w") as f:
        for r in rows:
            f.write(" ".join(r["title"].split()) + "\n")  # any line break or \r inside a title would shift every later line
    with open(os.path.join(OUT, "ids.txt"), "w") as f:
        f.write("\n".join(r["id"] for r in rows) + "\n")
    views = {}  # from fetch_details.py; videos without details (deleted, not fetched yet) get 0
    for p in glob.glob(os.path.join(D, "details", "*.jsonl")):
        views.update((d["id"], d["views"]) for d in map(json.loads, open(p)))
    np.array([min(views.get(r["id"], 0), 2**32 - 1) for r in rows], dtype="<u4").tofile(os.path.join(OUT, "views.bin"))
    print(f"views for {sum(r['id'] in views for r in rows):,}/{len(rows):,} videos", flush=True)
    islands = channel_islands(emb2, rec["ch"], cids, channels, rec["pad"])
    print(f"{len(islands)} channel islands; non-Minecraft videos: {int((np.array([r.get('mc', 1.0) for r in rows]) < 0.3).sum()):,}", flush=True)

    paths = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        paths[r["cid"]][month_of(r["date"]) // 3].append(i)
    meta_ch = []
    for c in cids:
        ch = channels[c]
        q = sorted((k, v) for k, v in paths[c].items() if len(v) >= 3)
        meta_ch.append({"id": c, "name": ch["name"], "handle": ch.get("handle", ""), "subs": ch["subs"],
                        "lang": ch.get("lang", ""), "n": sum(len(v) for v in paths[c].values()),
                        "path": [[k * 3, round(float(emb2[v, 0].mean()), 3), round(float(emb2[v, 1].mean()), 3),
                                  len(v)] for k, v in q]})
    json.dump({"channels": meta_ch, "clusters": clusters, "islands": islands, "n": len(rows),
               "bounds": [float(emb2[:, 0].min()), float(emb2[:, 1].min()),
                          float(emb2[:, 0].max()), float(emb2[:, 1].max())],
               "month0": "2005-01"}, open(os.path.join(OUT, "meta.json"), "w"), ensure_ascii=False)
    print(f"wrote {OUT}: {len(rows):,} videos, {len(cids)} channels, {len(clusters)} clusters")


def channel_islands(emb2, ch, cids, channels, game, k=30, min_n=120):
    """Where one channel is talking to itself: per video, the share of its k nearest map neighbours from the same
    channel -> web/data/islands.bin (u8, 0-255). Channels with min_n+ videos over 50% get a label at their median;
    mx/my place it among the island's Minecraft videos, for when other games are filtered out (no label: mx absent)."""
    from scipy.spatial import cKDTree
    _, nb = cKDTree(emb2).query(emb2, k=k + 1)
    same = (ch[nb[:, 1:]] == ch[:, None]).mean(1)
    np.round(same * 255).astype(np.uint8).tofile(os.path.join(OUT, "islands.bin"))

    def centre(idx):  # the island's densest part: points near the median, so a channel with two islands labels the bigger one
        med = np.median(emb2[idx], axis=0)
        d = np.linalg.norm(emb2[idx] - med, axis=1)
        return np.median(emb2[idx[d <= np.quantile(d, 0.5)]], axis=0)
    out = []
    for i in range(len(cids)):
        idx = np.flatnonzero((ch == i) & (same > 0.5))
        if len(idx) < min_n:
            continue
        cx, cy = centre(idx)
        o = {"ch": i, "name": channels[cids[i]]["name"], "n": int(len(idx)), "x": float(cx), "y": float(cy)}
        mc = idx[game[idx] == 0]
        if len(mc) >= min_n:
            o["mx"], o["my"] = map(float, centre(mc))
        out.append(o)
    return sorted(out, key=lambda o: -o["n"])


def main():
    channels, rows, X = load()
    print(f"{len(rows):,} videos from {len({r['cid'] for r in rows})} channels", flush=True)
    key = np.array([r["cid"] + r["id"] for r in rows])
    if "--apply" in sys.argv:
        st = np.load(STATE)
        assert np.array_equal(st["key"], key), "videos changed since the last full run; run without --apply"
        emb2, labels = st["emb2"], st["labels"]
    else:
        emb2, labels = layout(X, key, chan_codes(rows))
        np.savez(STATE, emb2=emb2, labels=labels, key=key)
    write(channels, rows, X, emb2, labels)


if __name__ == "__main__":
    main()
