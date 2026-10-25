"""Embed every thumbnail with Gemini Embedding 2 (image-only, 768 dims).

One data/emb_thumb/<cid>.npy (float16) per channel, rows aligned with data/videos/<cid>.jsonl;
a video with no thumbnail gets a zero row (map.py drops those). Incremental: finished channels are skipped.
Only channels in data/channels.json. 100 images per batchEmbedContents call, WORKERS channels at once.
  WORKERS=16 python3 embed_thumbs.py
"""
import base64, json, os, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import reuse

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "data")
EMB = os.path.join(D, "emb_thumb")
THUMBS = os.path.join(D, "thumbs")
DIMS, BATCH = 768, 100
WORKERS = int(os.environ.get("WORKERS", 16))
URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents"


def env():
    for line in open(os.path.join(ROOT, ".env")):
        k, _, v = line.strip().partition("=")
        if k:
            os.environ.setdefault(k, v)


def embed(paths):
    reqs = [{"model": "models/gemini-embedding-2", "output_dimensionality": DIMS,
             "content": {"parts": [{"inline_data": {"mime_type": "image/jpeg",
                                                    "data": base64.b64encode(open(p, "rb").read()).decode()}}]}}
            for p in paths]
    body = json.dumps({"requests": reqs}).encode()
    for attempt in range(8):
        req = urllib.request.Request(URL, data=body, headers={
            "x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return [e["values"] for e in json.load(r)["embeddings"]]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if attempt == 7 or (isinstance(e, urllib.error.HTTPError) and e.code in (400, 401, 403)):
                raise
            print("  retry:", getattr(e, "code", e), file=sys.stderr, flush=True)
            time.sleep(min(60, 2 ** attempt))


def do_channel(cid):
    vids = [json.loads(l)["id"] for l in open(os.path.join(D, "videos", cid + ".jsonl"))]
    out = np.zeros((len(vids), DIMS), dtype=np.float16)
    old = reuse.prior(cid, os.path.join(EMB, cid + ".npy"))
    for i, v in enumerate(vids):
        if v in old and np.any(old[v]):
            out[i] = old[v]
    have = [(i, os.path.join(THUMBS, v[:2], v + ".jpg")) for i, v in enumerate(vids) if not out[i].any()]
    have = [(i, p) for i, p in have if os.path.exists(p)]
    for s in range(0, len(have), BATCH):
        chunk = have[s:s + BATCH]
        out[[i for i, _ in chunk]] = np.asarray(embed([p for _, p in chunk]), dtype=np.float16)
    np.save(os.path.join(EMB, cid + ".npy.tmp.npy"), out)
    os.replace(os.path.join(EMB, cid + ".npy.tmp.npy"), os.path.join(EMB, cid + ".npy"))
    return len(have)


def main():
    env()
    os.makedirs(EMB, exist_ok=True)
    cids = [c["id"] for c in json.load(open(os.path.join(D, "channels.json")))]
    todo = [c for c in cids if os.path.exists(os.path.join(D, "videos", c + ".jsonl"))
            and reuse.todo(c, os.path.join(EMB, c + ".npy"))]
    print(f"{len(todo)} channels to embed, {WORKERS} workers", flush=True)
    n, t0 = 0, time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, k in enumerate(ex.map(do_channel, todo), 1):
            n += k
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} channels, {n:,} images, {n / (time.time() - t0):.0f}/s", flush=True)


if __name__ == "__main__":
    main()
