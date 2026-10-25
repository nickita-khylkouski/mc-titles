"""Keep what we already paid for when a channel's video list grows (fetch_videos.py REDO_CAPPED=1).
Per-channel outputs (data/emb/<cid>.npy, data/is_mc/<cid>.json, ...) are rows aligned with data/videos/<cid>.jsonl.
After a re-pull they're stale (row count differs); their rows still line up with data/videos_capped_backup/<cid>.jsonl,
so prior() maps video id -> old row and the embed scripts only pay for the new videos."""
import json, os

import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _rows(path):
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    return json.load(open(path))


def _count(path):
    return sum(1 for _ in open(path))


def todo(cid, path):
    """True if the output is missing or no longer lines up with the channel's video list."""
    if not os.path.exists(path):
        return True
    return len(_rows(path)) != _count(os.path.join(D, "videos", cid + ".jsonl"))


def prior(cid, path):
    """{video id: old row} from a stale output, or {} if there's nothing reusable."""
    bk = os.path.join(D, "videos_capped_backup", cid + ".jsonl")
    if not (os.path.exists(path) and os.path.exists(bk)):
        return {}
    rows = _rows(path)
    ids = [json.loads(l)["id"] for l in open(bk)]
    if len(rows) != len(ids):
        return {}
    return {v: rows[i] for i, v in enumerate(ids)}
