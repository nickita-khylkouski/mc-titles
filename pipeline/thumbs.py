"""Download every video's thumbnail (no API quota: thumbnails are plain CDN URLs keyed by video id).

mqdefault = 320x180, the true 16:9 frame (hqdefault adds black letterbox bars). ~12 KB each.
Saved as data/thumbs/<first 2 chars of id>/<id>.jpg. Resumable: existing files are skipped,
dead videos (404) are listed in data/thumbs/missing.txt so reruns don't retry them.
  WORKERS=64 python3 pipeline/thumbs.py
"""
import glob, http.client, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
OUT = os.path.join(D, "thumbs")
WORKERS = int(os.environ.get("WORKERS", 64))
MISSING = os.path.join(OUT, "missing.txt")
lock = threading.Lock()
local = threading.local()  # one keep-alive connection per thread: skips a TLS handshake per image


def path(vid):
    return os.path.join(OUT, vid[:2], vid + ".jpg")


def get(vid):
    for attempt in range(4):
        try:
            if not hasattr(local, "c"):
                local.c = http.client.HTTPSConnection("i.ytimg.com", timeout=20)
            local.c.request("GET", f"/vi/{vid}/mqdefault.jpg")
            r = local.c.getresponse()
            data = r.read()
            if r.status == 404:
                with lock, open(MISSING, "a") as f:
                    f.write(vid + "\n")
                return "missing"
            if r.status != 200:
                raise OSError(r.status)  # 429/5xx: back off
            p = path(vid)
            with open(p + ".tmp", "wb") as f:
                f.write(data)
            os.replace(p + ".tmp", p)
            return "ok"
        except Exception:
            if hasattr(local, "c"):
                local.c.close()
                del local.c
            time.sleep(2 ** attempt)
    return "failed"


def main():
    os.makedirs(OUT, exist_ok=True)
    missing = set(open(MISSING).read().split()) if os.path.exists(MISSING) else set()
    vids = []
    for p in sorted(glob.glob(os.path.join(D, "videos", "*.jsonl"))):
        vids += [json.loads(l)["id"] for l in open(p)]
    for v in {v[:2] for v in vids}:
        os.makedirs(os.path.join(OUT, v), exist_ok=True)
    todo = [v for v in dict.fromkeys(vids) if v not in missing and not os.path.exists(path(v))]
    print(f"{len(vids):,} videos, {len(todo):,} thumbnails to fetch with {WORKERS} workers", flush=True)
    counts, t0 = {"ok": 0, "missing": 0, "failed": 0}, time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, res in enumerate(ex.map(get, todo), 1):
            counts[res] += 1
            if i % 5000 == 0 or i == len(todo):
                rate = i / (time.time() - t0)
                print(f"  {i:,}/{len(todo):,}  {counts}  {rate:.0f}/s  eta {(len(todo) - i) / rate / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
