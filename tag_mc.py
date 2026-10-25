"""Per-video "is this Minecraft?" with the Decisions API (gpt-6-luna): 100 numbered titles from one channel per
call (same-channel context helps with titles like "Episode 41"), one predicate per title.
Writes data/is_mc/<cid>.json: probabilities aligned with data/videos/<cid>.jsonl. map.py hides videos under
0.3 behind the "Minecraft only" toggle, same as other-game clusters. Resumable per channel.
  WORKERS=3 python3 tag_mc.py
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor

from classify import env
from decide import decide, predicate
import reuse

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "data")
OUT = os.path.join(D, "is_mc")
BATCH = 100
Q = "Is video {i} about Minecraft (gameplay, mods, servers, Minecraft animations/memes/Shorts)? Other games, comics, vlogs = no."


def one(c):
    path = os.path.join(OUT, c["id"] + ".json")
    vids = [json.loads(l) for l in open(os.path.join(D, "videos", c["id"] + ".jsonl"))]
    old = reuse.prior(c["id"], path)
    new = [i for i, v in enumerate(vids) if v["id"] not in old]
    titles = [vids[i]["title"].replace("\n", " ") for i in new]
    p = []
    for s in range(0, len(titles), BATCH):
        chunk = titles[s:s + BATCH]
        text = f"Titles from the YouTube channel {c['name']}, numbered:\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(chunk))
        a = decide(text, [predicate(f"v{i}", Q.format(i=i)) for i in range(len(chunk))])
        p += [round(float(a[f"v{i}"]), 3) for i in range(len(chunk))]
    got = dict(zip(new, p))
    p = [got[i] if i in got else old[v["id"]] for i, v in enumerate(vids)]
    json.dump(p, open(path + ".tmp", "w"))
    os.replace(path + ".tmp", path)
    return c["name"], sum(x < 0.3 for x in p), len(p)


def main():
    env()
    os.makedirs(OUT, exist_ok=True)
    chs = [c for c in json.load(open(os.path.join(D, "channels.json")))
           if os.path.exists(os.path.join(D, "videos", c["id"] + ".jsonl")) and reuse.todo(c["id"], os.path.join(OUT, c["id"] + ".json"))]
    print(len(chs), "channels to tag", flush=True)

    def safe(c):
        try:
            return one(c)
        except Exception as e:
            print(f"FAILED {c['name']}: {e!r}", flush=True)
    done = 0
    for r in ThreadPoolExecutor(int(os.environ.get("WORKERS", 3))).map(safe, chs):
        done += 1
        if r and (r[1] / max(r[2], 1) > 0.3 or done % 50 == 0):
            print(f"  [{done}/{len(chs)}] {r[0]}: {r[1]}/{r[2]} not Minecraft", flush=True)


if __name__ == "__main__":
    main()
