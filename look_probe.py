"""One linear probe per thumbnail look, trained on label_thumbs.py's labels over the 768-d Gemini thumbnail vectors.
5-fold cross-validation gives each look's held-out AUC and accuracy; looks the probe can't learn (AUC < 0.8) are
flagged so analyze.py skips them. Saved per look: [w (768), b, cutoff] on raw unit vectors. Writes data/look_probes.npz (+ a printed report).
  nice uv run python look_probe.py
"""
import glob, json, os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "data")


def thumb_vecs(ids):
    """id -> unit thumbnail vector, straight from data/emb_thumb (independent of any map build)."""
    want, out = set(ids), {}
    for p in glob.glob(os.path.join(D, "emb_thumb", "*.npy")):
        cid = os.path.basename(p)[:-4]
        vp = os.path.join(D, "videos", cid + ".jsonl")
        if not os.path.exists(vp):
            continue
        vids = [json.loads(l)["id"] for l in open(vp)]
        e = np.load(p, mmap_mode="r")
        for i, v in enumerate(vids):
            if v in want and np.any(e[i]):
                x = e[i].astype(np.float32)
                out[v] = x / np.linalg.norm(x)
    return out


def main():
    L = [json.loads(l) for l in open(os.path.join(D, "thumb_labels.jsonl"))]
    vec = thumb_vecs([l["id"] for l in L])
    L = [l for l in L if l["id"] in vec]
    X = np.stack([vec[l["id"]] for l in L])
    looks = [k for k in L[0] if k != "id"]
    print(f"{len(L)} labelled thumbnails")
    save = {}
    for k in looks:
        y = np.array([l[k] for l in L])
        if y.sum() < 40 or (1 - y).sum() < 40:
            print(f"  {k:16s} too rare ({y.mean():.1%}), skipped")
            continue
        # unit vectors give tiny features: standardise, pick C by held-out log-loss, then fold the scaler into w, b
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - mu) / sd
        best = None
        for C in (0.003, 0.01, 0.03, 0.1):
            p = np.zeros(len(y))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Z, y):
                p[te] = LogisticRegression(C=C, max_iter=3000).fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
            ll = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
            if best is None or ll < best[0]:
                best = (ll, C, p)
        _, C, p = best
        auc = roc_auc_score(y, p)
        # cutoff: the one with the best F1 on held-out predictions
        cuts = np.quantile(p, np.linspace(0.5, 0.99, 50))
        f1 = [2 * (y[p >= c].mean() * (p[y == 1] >= c).mean()) / (y[p >= c].mean() + (p[y == 1] >= c).mean() + 1e-9) for c in cuts]
        cut = float(cuts[int(np.argmax(f1))])
        hit = p >= cut
        prec, rec, acc = y[hit].mean(), hit[y == 1].mean(), (hit == y).mean()
        m = LogisticRegression(C=C, max_iter=3000).fit(Z, y)
        w = m.coef_[0] / sd
        b = m.intercept_[0] - (m.coef_[0] * mu / sd).sum()
        save[k] = np.concatenate([w, [b, cut]]).astype(np.float32)
        save[k + "__stats"] = np.array([auc, acc, prec, rec, y.mean(), hit.mean()], dtype=np.float32)
        print(f"  {k:16s} share {y.mean():5.1%}  C {C:<5}  held-out AUC {auc:.3f}  at cutoff: precision {prec:.0%} recall {rec:.0%} "
              f"(flags {hit.mean():.1%}){'' if auc >= 0.8 and prec >= 0.6 else '   <- too weak, not used'}")
    np.savez(os.path.join(D, "look_probes.npz"), **save)


if __name__ == "__main__":
    main()
