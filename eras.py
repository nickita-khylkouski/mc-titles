"""Era summary from web/data (MAP_SRC=rich|thumb for the other maps): per year, the Minecraft clusters most
over-represented vs. all years (lift), by uploads and, when views.bin has data, by views.
Views lift = the cluster's share of that year's views over its share of all views: what people actually watched."""
import json, os
import numpy as np

SRC = os.environ.get("MAP_SRC", "title")
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "data" + ("" if SRC == "title" else "_" + SRC))
meta = json.load(open(os.path.join(W, "meta.json")))
r = np.fromfile(os.path.join(W, "points.bin"), dtype=[("x", "<f4"), ("y", "<f4"), ("ch", "<u2"), ("m", "<u2"),
                                                      ("cl", "<i2"), ("kind", "u1"), ("game", "u1")])
vp = os.path.join(W, "views.bin")
views = np.fromfile(vp, dtype="<u4").astype(np.float64) if os.path.exists(vp) else np.zeros(len(r))
has_views = len(views) == len(r) and (views > 0).mean() > 0.5
names = {c["id"]: c["name"] for c in meta["clusters"]}
nc = max(names) + 1
keep = r["game"] == 0
mc, mv = r[keep], views[keep] if has_views else None
year = mc["m"] // 12 + 2005
cl = mc["cl"]


def tops(sel, w, all_share, min_n):
    """top-5 clusters by lift in `sel`; share = weighted count / total weight in sel (unclustered included)"""
    ww = np.ones(sel.sum()) if w is None else w[sel]
    c = cl[sel]
    cnt = np.bincount(c[c >= 0], minlength=nc)
    share = np.bincount(c[c >= 0], weights=ww[c >= 0], minlength=nc) / max(ww.sum(), 1)
    lift = share / np.maximum(all_share, 1e-12)
    return [k for k in np.argsort(-lift) if cnt[k] >= min_n][:5], lift


all_up = np.bincount(cl[cl >= 0], minlength=nc) / len(mc)
all_v = np.bincount(cl[cl >= 0], weights=mv[cl >= 0], minlength=nc) / mv.sum() if has_views else None  # missing views weigh 0
for y in range(2010, 2027):
    sel = year == y
    if sel.sum() < 500:
        continue
    top, lift = tops(sel, None, all_up, 40)
    short = (mc["kind"][sel] == 1).mean()
    print(f"{y} ({sel.sum():,} videos, {short:.0%} Shorts)" + (f", {mv[sel].sum() / 1e9:.1f}B views" if has_views else ""))
    print("   uploads: " + ", ".join(f"{names[k]} ×{lift[k]:.0f}" for k in top))
    if has_views:
        top, lift = tops(sel, mv, all_v, 40)
        print("   views:   " + ", ".join(f"{names[k]} ×{lift[k]:.0f}" for k in top))
g = r["game"] == 1
print(f"\nother-game clusters: {g.sum():,} videos ({g.mean():.1%}); peak year",
      np.bincount((r['m'][g] // 12 + 2005)).argmax())
if has_views:
    # clusters that punch above their weight: median views per video vs the median video, only videos whose
    # views we have (missing = not fetched yet, not 0 views); a cluster needs 300 such videos
    known = mv > 0
    med = np.median(mv[known])
    ratio, big = np.zeros(nc), []
    for k in range(nc):
        vk = mv[known & (cl == k)]
        if len(vk) >= 300:
            ratio[k] = np.median(vk) / med
            big.append(k)
    big.sort(key=lambda k: -ratio[k])
    print(f"\nviews known for {known.mean():.0%} of videos")
    print("most watched per video (median):", ", ".join(f"{names[k]} ×{ratio[k]:.1f}" for k in big[:10]))
    print("least watched per video (median):", ", ".join(f"{names[k]} ×{ratio[k]:.2f}" for k in big[-10:]))
