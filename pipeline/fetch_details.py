"""Step 2b: per-video details the playlist pull doesn't give: tags, category, views/likes/comments, duration.

videos.list costs 1 quota unit per 50 ids whatever parts are asked for, so ~7.3k units for all videos (one day).
`fields` trims the response to just what we keep (no descriptions, those came with the playlist pull).
One data/details/<cid>.jsonl per channel; finished channels are skipped. Deleted videos just don't come back.
A channel whose videos file is newer than its details (re-pulled uncapped) only fetches the ids it lacks.
  WORKERS=2 python3 pipeline/fetch_details.py
"""
import json, os, re, sys, threading, urllib.parse
from concurrent.futures import ThreadPoolExecutor

from fetch_videos import D, QuotaDone, env, lock, stop
import fetch_videos as fv

OUT = os.path.join(D, "details")
WORKERS = int(os.environ.get("WORKERS", 2))
FIELDS = ("items(id,snippet(tags,categoryId,defaultAudioLanguage),"
          "statistics(viewCount,likeCount,commentCount),contentDetails(duration))")


def seconds(iso):
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    if not m:
        return 0
    d, h, mi, s = (int(x or 0) for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + s


def call(ids):
    import time, urllib.error, urllib.request
    with lock:
        if stop.is_set() or fv.used >= fv.QUOTA_BUDGET:
            stop.set()
            raise QuotaDone
        fv.used += 1
    url = "https://www.googleapis.com/youtube/v3/videos?" + urllib.parse.urlencode({
        "part": "snippet,statistics,contentDetails", "id": ",".join(ids), "fields": FIELDS,
        "key": os.environ["YOUTUBE_API_KEY"]})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r).get("items", [])
        except urllib.error.HTTPError as e:
            if "quotaExceeded" in e.read().decode(errors="replace"):
                stop.set()
                raise QuotaDone
            if attempt == 4:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 4:
                raise
        time.sleep(2 ** attempt)


def one(cid, done=[0]):
    path = os.path.join(OUT, cid + ".jsonl")
    vpath = os.path.join(D, "videos", cid + ".jsonl")
    rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    have = {r["id"] for r in rows}
    ids = [i for i in (json.loads(l)["id"] for l in open(vpath)) if i not in have]
    try:
        fetch(ids, rows)
    except QuotaDone:  # keep what we got, but leave the file older than the video list so the next run resumes it
        save(path, rows)
        t = os.path.getmtime(vpath) - 1
        os.utime(path, (t, t))
        raise
    save(path, rows)
    with lock:
        done[0] += 1
        if done[0] % 20 == 0:
            print(f"  {done[0]} channels, quota used {fv.used}", flush=True)


def save(path, rows):
    with open(path + ".tmp", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    os.replace(path + ".tmp", path)


def fetch(ids, rows):
    for s in range(0, len(ids), 50):
        for it in call(ids[s:s + 50]):
            sn, st = it.get("snippet", {}), it.get("statistics", {})
            rows.append({"id": it["id"], "tags": sn.get("tags", []), "cat": sn.get("categoryId"),
                         "lang": sn.get("defaultAudioLanguage"), "views": int(st.get("viewCount", 0)),
                         "likes": int(st.get("likeCount", 0)), "comments": int(st.get("commentCount", 0)),
                         "dur": seconds(it.get("contentDetails", {}).get("duration"))})


def main():
    env()
    os.makedirs(OUT, exist_ok=True)
    cids = [c["id"] for c in json.load(open(os.path.join(D, "channels.json")))]
    vp, dp = (lambda c: os.path.join(D, "videos", c + ".jsonl")), (lambda c: os.path.join(OUT, c + ".jsonl"))
    todo = [c for c in cids if os.path.exists(vp(c))
            and (not os.path.exists(dp(c)) or os.path.getmtime(dp(c)) < os.path.getmtime(vp(c)))]
    todo.sort(key=lambda c: os.path.getsize(vp(c)))
    print(f"{len(todo)} channels to fetch, {WORKERS} workers", flush=True)

    def safe(c):
        try:
            one(c)
        except QuotaDone:
            return "quota"
        except Exception as e:
            print(f"FAILED {c}: {e!r}", flush=True)
    res = list(ThreadPoolExecutor(WORKERS).map(safe, todo))
    print(f"done, quota used {fv.used}" + ("; quota reached, rerun tomorrow" if "quota" in res else ""), flush=True)
    sys.exit(2 if "quota" in res else 0)


if __name__ == "__main__":
    main()
