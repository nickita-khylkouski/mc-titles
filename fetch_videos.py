"""Step 2: pull every upload (title, date, description) for the channels in data/channels.json.

Uses the channel's long-form (UULF…) and Shorts (UUSH…) upload playlists, so each video is tagged
long/short and live-stream VODs stay out. 1 quota unit per 50 videos.

Resumable: one file per channel in data/videos/<cid>.jsonl; finished channels are skipped.
Stops cleanly before the daily quota (QUOTA_BUDGET) or on a quotaExceeded error; rerun tomorrow.
Channels over CAP videos keep CAP evenly spaced across their whole history, not just the newest.
CAP=0 means no cap. REDO_CAPPED=1 re-pulls channels whose file hit the old 1500 cap (old file copied to
data/videos_capped_backup/ first; finished ones are listed in data/uncapped.txt so reruns skip them).
  CAP=0 REDO_CAPPED=1 QUOTA_BUDGET=8000 python3 fetch_videos.py
"""
import json, os, sys, threading, time, urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "data")
OUT = os.path.join(D, "videos")
CAP = int(os.environ.get("CAP", 1500))
REDO = os.environ.get("REDO_CAPPED") == "1"
BACKUP = os.path.join(D, "videos_capped_backup")
UNCAPPED = os.path.join(D, "uncapped.txt")
QUOTA_BUDGET = int(os.environ.get("QUOTA_BUDGET", 9500))
WORKERS = int(os.environ.get("WORKERS", 4))
used = 0
lock = threading.Lock()
stop = threading.Event()


class QuotaDone(Exception):
    pass


def env():
    for line in open(os.path.join(ROOT, ".env")):
        k, _, v = line.strip().partition("=")
        if k:
            os.environ.setdefault(k, v)


def page(playlist, token=None):
    global used
    with lock:
        if stop.is_set() or used >= QUOTA_BUDGET:
            stop.set()
            raise QuotaDone
        used += 1
    params = {"part": "snippet,contentDetails", "maxResults": 50, "playlistId": playlist,
              "key": os.environ["YOUTUBE_API_KEY"]}
    if token:
        params["pageToken"] = token
    url = "https://www.googleapis.com/youtube/v3/playlistItems?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if e.code == 404:          # channel has no Shorts / no long-form playlist
                return {"items": []}
            if "quotaExceeded" in body:
                stop.set()
                raise QuotaDone
            if attempt == 3:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)


def pull(playlist, kind):
    vids, token = [], None
    while True:
        d = page(playlist, token)
        for it in d.get("items", []):
            sn = it["snippet"]
            if sn.get("title") in ("Private video", "Deleted video"):
                continue
            vids.append({
                "id": it["contentDetails"]["videoId"],
                "date": it["contentDetails"].get("videoPublishedAt") or sn.get("publishedAt"),
                "title": sn["title"],
                "desc": sn.get("description") or "",
                "kind": kind,
            })
        token = d.get("nextPageToken")
        if not token:
            return vids


def even_sample(vids, n):
    if not n or len(vids) <= n:
        return vids
    vids = sorted(vids, key=lambda v: v["date"])
    step = len(vids) / n
    return [vids[int(i * step)] for i in range(n)]


def main():
    env()
    os.makedirs(OUT, exist_ok=True)
    channels = json.load(open(os.path.join(D, "channels.json")))
    todo = [c for c in channels if not os.path.exists(os.path.join(OUT, c["id"] + ".jsonl"))]
    if REDO:
        import shutil
        os.makedirs(BACKUP, exist_ok=True)
        fin = set(open(UNCAPPED).read().split()) if os.path.exists(UNCAPPED) else set()
        for c in channels:
            p = os.path.join(OUT, c["id"] + ".jsonl")
            if c["id"] in fin or not os.path.exists(p) or sum(1 for _ in open(p)) < 1500:
                continue
            if not os.path.exists(os.path.join(BACKUP, c["id"] + ".jsonl")):
                shutil.copy2(p, BACKUP)
            todo.append(c)
        todo.sort(key=lambda c: c.get("videos") or 0)  # small ones first: a quota stop mid-channel wastes less
        print(f"{len(todo)} channels to (re)pull uncapped", flush=True)
    done = len(channels) - len(todo)

    def one(c):
        nonlocal done
        path = os.path.join(OUT, c["id"] + ".jsonl")
        rest = c["id"][2:]
        vids = even_sample(pull("UULF" + rest, "long") + pull("UUSH" + rest, "short"), CAP)
        with open(path + ".tmp", "w") as f:
            for v in vids:
                f.write(json.dumps({"cid": c["id"], **v}, ensure_ascii=False) + "\n")
        os.replace(path + ".tmp", path)
        with lock:
            if REDO:
                open(UNCAPPED, "a").write(c["id"] + "\n")
            done += 1
            print(f"[{done}/{len(channels)}] {c['name']}: {len(vids)} videos  (quota used {used})", flush=True)

    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(lambda c: _safe(one, c), todo))
    if any(r == "quota" for r in results):
        print(f"quota budget reached at {used} units; {done}/{len(channels)} channels done. rerun tomorrow.")
        sys.exit(2)
    print(f"all {len(channels)} channels done, quota used {used}")


def _safe(fn, c):
    try:
        fn(c)
    except QuotaDone:
        return "quota"
    except Exception as e:  # one bad channel shouldn't kill the run; it stays undone and retries next run
        print(f"FAILED {c['name']}: {e!r}", flush=True)
        return "error"

if __name__ == "__main__":
    main()
