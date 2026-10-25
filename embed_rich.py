"""Step 3b: embed title + cleaned description (OpenAI text-embedding-3-small, 512-d) per video.

The description is cut to the part that is about this video (clean_desc.py drops the channel's repeated
boilerplate). Tags are left out: about half of them are channel-wide SEO lists and 15% of videos have none.
Writes data/emb_rich/<cid>.npy, rows aligned with data/videos/<cid>.jsonl. Map it with MAP_SRC=rich.
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import reuse

from clean_desc import cleaner
from embed import D, DIMS, embed, env

EMB = os.path.join(D, "emb_rich")
BATCH = 400


def text_of(v, clean):
    t = v["title"].strip() or "(untitled)"
    d = clean(v["desc"])
    return t + ("\n" + d if d else "")


def do_channel(cid):
    vids = [json.loads(l) for l in open(os.path.join(D, "videos", cid + ".jsonl"))]
    clean = cleaner([v["desc"] for v in vids])
    old = reuse.prior(cid, os.path.join(EMB, cid + ".npy"))  # old rows were cleaned against the capped list; close enough
    out = np.zeros((len(vids), DIMS), dtype=np.float16)
    new = [i for i, v in enumerate(vids) if v["id"] not in old]
    for i, v in enumerate(vids):
        if v["id"] in old:
            out[i] = old[v["id"]]
    for s in range(0, len(new), BATCH):
        out[new[s:s + BATCH]] = np.asarray(embed([text_of(vids[i], clean) for i in new[s:s + BATCH]]), dtype=np.float16)
    np.save(os.path.join(EMB, cid + ".tmp.npy"), out)
    os.replace(os.path.join(EMB, cid + ".tmp.npy"), os.path.join(EMB, cid + ".npy"))
    return len(new)


def main():
    env()
    os.makedirs(EMB, exist_ok=True)
    cids = [c["id"] for c in json.load(open(os.path.join(D, "channels.json")))]
    todo = [c for c in cids if os.path.exists(os.path.join(D, "videos", c + ".jsonl"))
            and reuse.todo(c, os.path.join(EMB, c + ".npy"))]
    print(len(todo), "channels to embed", flush=True)
    n = 0
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 3))) as ex:
        for i, k in enumerate(ex.map(do_channel, todo), 1):
            n += k
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} channels, {n:,} videos", flush=True)


if __name__ == "__main__":
    main()
