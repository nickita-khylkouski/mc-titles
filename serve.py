"""The UI server: web/ as static files plus two small vector endpoints (local only, :8395).

  /api/search?map=&q=   embed the phrase in the map's own space (OpenAI for titles/words, Gemini text for
                        thumbnails: Gemini Embedding 2 puts text and images in one space) -> closest videos
  /api/nn?map=&i=       a video's true nearest neighbours in the full vector space, how far each one ended
                        up on the 2-d map (rank by map distance), and the vector itself (for the barcode)

Vectors come from data/vecs{_rich,_thumb}.npy (written by map.py, same order as points.bin). One map is kept
in memory at a time. Query embeddings are cached in data/query_cache.json, so a repeat search is offline.
  uv run python serve.py
"""
import json, os, sys, threading, urllib.parse, urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "data")
WEB = os.path.join(ROOT, "web")
CACHE = os.path.join(D, "query_cache.json")
SFX = {"": "", "title": "", "rich": "_rich", "thumb": "_thumb"}
lock = threading.Lock()
loaded = {}  # sfx -> (vecs float32, xy float32)
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def env():
    for line in open(os.path.join(ROOT, ".env")):
        k, _, v = line.strip().partition("=")
        if k:
            os.environ.setdefault(k, v)


def get(sfx):
    with lock:
        if sfx not in loaded:
            loaded.clear()  # one map in memory at a time (~0.6-0.9 GB)
            v = np.load(os.path.join(D, f"vecs{sfx}.npy")).astype(np.float32)
            pts = np.fromfile(os.path.join(WEB, "data" + sfx, "points.bin"), dtype=[("x", "<f4"), ("y", "<f4"), ("r", "V8")])
            assert len(pts) == len(v), "vecs and points.bin differ; rerun map.py --apply"
            loaded[sfx] = (v, np.stack([pts["x"], pts["y"]], 1))
        return loaded[sfx]


def post(url, body, headers):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def embed_query(sfx, q):
    key = f"{sfx}|{q}"
    if key not in cache:
        if sfx == "_thumb":
            r = post("https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent",
                     {"model": "models/gemini-embedding-2", "output_dimensionality": 768, "content": {"parts": [{"text": q}]}},
                     {"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
            e = r["embedding"]["values"]
        else:
            r = post("https://api.openai.com/v1/embeddings", {"model": "text-embedding-3-small", "input": [q], "dimensions": 512},
                     {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]})
            e = r["data"][0]["embedding"]
        with lock:
            cache[key] = [round(x, 5) for x in e]
            json.dump(cache, open(CACHE + ".tmp", "w"))
            os.replace(CACHE + ".tmp", CACHE)
    e = np.array(cache[key], dtype=np.float32)
    return e / (np.linalg.norm(e) + 1e-8)


def search(sfx, q, n=3000):
    v, xy = get(sfx)
    s = v @ embed_query(sfx, q)
    top = np.argpartition(-s, n)[:n]
    top = top[np.argsort(-s[top])]
    # a typical score, so the UI can say how strong the best match is compared with a random video
    base = float(np.median(s[:: max(1, len(s) // 20000)]))
    return {"idx": top.tolist(), "sim": np.round(s[top], 4).tolist(), "median": round(base, 4)}


def nn(sfx, i, k=12):
    v, xy = get(sfx)
    s = v @ v[i]
    s[i] = -9
    top = np.argpartition(-s, k)[:k]
    top = top[np.argsort(-s[top])]
    d2 = ((xy - xy[i]) ** 2).sum(1)
    rank = np.searchsorted(np.sort(d2), d2[top])  # 1 = the closest dot on the map
    # how much of the true neighbourhood the 2-d map kept: of these k, how many sit in the closest 1% of dots
    near = int((rank < len(xy) // 100).sum())
    q = np.clip(np.round(v[i] / (np.abs(v[i]).max() + 1e-8) * 127), -127, 127).astype(int)
    return {"idx": top.tolist(), "sim": np.round(s[top], 4).tolist(), "maprank": rank.tolist(), "near1": near, "n": len(xy),
            "dims": int(v.shape[1]), "vec": q.tolist(), "scale": round(float(np.abs(v[i]).max()), 5),
            "nvec": [np.clip(np.round(v[j] / (np.abs(v[j]).max() + 1e-8) * 127), -127, 127).astype(int).tolist() for j in top[:3]]}


class H(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if not u.path.startswith("/api/"):
            return super().do_GET()
        p = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        try:
            sfx = SFX[p.get("map", "")]
            if u.path == "/api/search":
                out = search(sfx, p["q"].strip()[:200])
            elif u.path == "/api/nn":
                out = nn(sfx, int(p["i"]))
            else:
                return self.send_error(404)
            body, code = json.dumps(out).encode(), 200
        except Exception as e:
            body, code = json.dumps({"error": repr(e)}).encode(), 500
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    env()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8395
    ThreadingHTTPServer(("127.0.0.1", port), partial(H, directory=WEB)).serve_forever()
