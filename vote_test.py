"""How far can the neighbour vote go? Extra rounds at looser `need` on the written title map; cohesion = cos to centroid."""
import json, numpy as np
import knn as K
PT = [("x","<f4"),("y","<f4"),("ch","<u2"),("m","<u2"),("cl","<i2"),("kind","u1"),("game","u1")]
r = np.fromfile("web/data/points.bin", dtype=PT); N = len(r)
meta = json.load(open("web/data/meta.json"))
ids = open("web/data/ids.txt").read().split("\n")[:N]
key = np.array([meta["channels"][c]["id"] + i for c, i in zip(r["ch"], ids)])
z = np.load("data/map_state.npz", allow_pickle=True); fkey = z["key"]
pos = {k: j for j, k in enumerate(fkey)}
full = np.array([pos[k] for k in key])
idx, sim = np.load("data/knn.npz")["idx"], np.load("data/knn.npz")["sim"].astype(np.float32)
chf = np.unique([k[:24] for k in fkey], return_inverse=True)[1]
inv = np.full(len(fkey), -1); inv[full] = np.arange(N)
other = K.cross(idx, sim, chf, other=16, same=0)[0][full, 1:]
other = inv[other]
X = np.load("data/vecs.npy").astype(np.float32)
cl = r["cl"].astype(np.int32)
C = np.zeros((cl.max() + 1, X.shape[1]), np.float32); np.add.at(C, cl[cl >= 0], X[cl >= 0]); C /= np.linalg.norm(C, axis=1, keepdims=True)
core = (X[cl >= 0] * C[cl[cl >= 0]]).sum(1).mean()
print(f"unclustered {np.mean(cl < 0):.3f}  member cos {core:.3f}")
for need in (7, 6, 5, 4):
    c2 = K.vote(cl, other, need=need, min_votes=need, rounds=4)
    new = (cl < 0) & (c2 >= 0)
    cos = (X[new] * C[c2[new]]).sum(1)
    # second-best centroid margin: is it really this cluster?
    s = X[new] @ C.T; s.sort(1)
    print(f"need {need}: unclustered {np.mean(c2 < 0):.3f}  new {new.mean():.3f} cos {cos.mean():.3f}  margin {np.median(s[:,-1]-s[:,-2]):.3f}")
lone = cl < 0
s = X[lone] @ C.T; best = s.max(1); s.sort(1)
print(f"still-unclustered cos to best centroid {best.mean():.3f} margin {np.median(s[:,-1]-s[:,-2]):.3f}")
m = X[cl >= 0] @ C.T; m.sort(1); print(f"members margin {np.median(m[:,-1]-m[:,-2]):.3f}")
