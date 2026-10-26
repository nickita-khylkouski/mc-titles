"""Every video's 12 closest videos, written as small files so the public (serverless) map can show them on click.
Exact: each row is split into blocks of 1,024, and the true top 12 always sit in the 13 blocks with the highest maxima.
  nice uv run python pipeline/static_nn.py [title|rich|thumb ...]   -> data/static_nn{sfx}/<chunk>.bin
Each chunk holds CHUNK videos: uint32 indices [CHUNK x 12], then uint8 similarity x 255 [CHUNK x 12].
"""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K, B, R, CHUNK = 12, 1024, 256, 2000
SFX = {"title": "", "rich": "_rich", "thumb": "_thumb"}

for name in sys.argv[1:] or list(SFX):
    sfx = SFX[name]
    v = np.load(os.path.join(ROOT, "data", f"vecs{sfx}.npy")).astype(np.float32)
    n, nb = len(v), -(-len(v) // B)
    out = os.path.join(ROOT, "data", f"static_nn{sfx}")
    os.makedirs(out, exist_ok=True)
    s = np.full((R, nb * B), -9, np.float32)
    I, S = np.zeros((n, K), np.uint32), np.zeros((n, K), np.uint8)
    t = time.time()
    for a in range(0, n, R):
        b = min(n, a + R)
        m = b - a
        np.matmul(v[a:b], v.T, out=s[:m, :n])
        s[np.arange(m), np.arange(a, b)] = -9  # not itself
        sb = s[:m].reshape(m, nb, B)
        tb = np.argpartition(-sb.max(2), K + 1, axis=1)[:, :K + 1]
        cand = np.take_along_axis(sb, tb[:, :, None], 1).reshape(m, -1)
        ci = (tb[:, :, None] * B + np.arange(B)).reshape(m, -1)
        p = np.argpartition(-cand, K, axis=1)[:, :K]
        idx, sim = np.take_along_axis(ci, p, 1), np.take_along_axis(cand, p, 1)
        o = np.argsort(-sim, 1)
        I[a:b], S[a:b] = np.take_along_axis(idx, o, 1), np.clip(np.round(np.take_along_axis(sim, o, 1) * 255), 0, 255)
        if a // R % 200 == 0:
            print(f"{name}: {b:,}/{n:,}  {time.time() - t:.0f}s", flush=True)
    for c in range(0, n, CHUNK):
        with open(os.path.join(out, f"{c // CHUNK}.bin"), "wb") as f:
            f.write(I[c:c + CHUNK].tobytes()); f.write(S[c:c + CHUNK].tobytes())
    print(f"{name}: wrote {-(-n // CHUNK)} chunks to {out} in {time.time() - t:.0f}s", flush=True)
