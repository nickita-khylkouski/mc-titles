"""Step 3: embed every fetched title with OpenAI text-embedding-3-small (512 dims).

Incremental: one data/emb/<cid>.npy (float16) per finished channel in data/videos/, row order matches
the .jsonl. Rerun any time; only channels without an .npy get embedded.
"""
import glob, json, os, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import reuse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
EMB = os.path.join(D, "emb")
MODEL, DIMS, BATCH = "text-embedding-3-small", 512, 1000


def env():
    for line in open(os.path.join(ROOT, ".env")):
        k, _, v = line.strip().partition("=")
        if k:
            os.environ.setdefault(k, v)


def embed(texts):
    body = json.dumps({"model": MODEL, "input": texts, "dimensions": DIMS}).encode()
    for attempt in range(10):
        req = urllib.request.Request("https://api.openai.com/v1/embeddings", data=body, headers={
            "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)["data"]
            return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if attempt == 9:
                raise
            print("  retry:", e, file=sys.stderr)
            time.sleep(min(60, 2 ** attempt))


def do_channel(path):
    cid = os.path.basename(path)[:-6]
    vids = [json.loads(l) for l in open(path)]
    titles = [v["title"].strip() or "(untitled)" for v in vids]
    old = reuse.prior(cid, os.path.join(EMB, cid + ".npy"))
    out = np.zeros((len(vids), DIMS), dtype=np.float16)
    new = [i for i, v in enumerate(vids) if v["id"] not in old]
    for i, v in enumerate(vids):
        if v["id"] in old:
            out[i] = old[v["id"]]
    for s in range(0, len(new), BATCH):
        out[new[s:s + BATCH]] = np.asarray(embed([titles[i] for i in new[s:s + BATCH]]), dtype=np.float16)
    np.save(os.path.join(EMB, cid + ".tmp.npy"), out)
    os.replace(os.path.join(EMB, cid + ".tmp.npy"), os.path.join(EMB, cid + ".npy"))
    return len(new)


def main():
    env()
    os.makedirs(EMB, exist_ok=True)
    todo = [p for p in sorted(glob.glob(os.path.join(D, "videos", "*.jsonl")))
            if reuse.todo(os.path.basename(p)[:-6], os.path.join(EMB, os.path.basename(p)[:-6] + ".npy"))]
    print(len(todo), "channels to embed", flush=True)
    n = 0
    with ThreadPoolExecutor(4) as ex:
        for i, k in enumerate(ex.map(do_channel, todo), 1):
            n += k
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} channels, {n:,} new titles", flush=True)


if __name__ == "__main__":
    main()
