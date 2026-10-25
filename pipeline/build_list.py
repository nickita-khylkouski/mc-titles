"""Step 1: build the candidate channel list.

Sources (all in data/):
  seed_mcmap.json         hand-picked MCYT list from the-minecraft-map repo (name -> channel id)
  hashtag_minecraft.txt   yt-dlp scrape of youtube.com/hashtag/minecraft  (cid|channel|vid|title)
  search_*.txt            yt-dlp scrapes of Minecraft searches sorted by views (same format)
  extra_handles.txt       one @handle per line for big names missing from the above

Writes data/candidates.json sorted by subs, with where each channel was found.
"""
import glob, json, os, sys, urllib.parse, urllib.request
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")


def env():
    for line in open(os.path.join(ROOT, ".env")):
        k, _, v = line.strip().partition("=")
        if k:
            os.environ.setdefault(k, v)


def yt(endpoint, **params):
    params["key"] = os.environ["YOUTUBE_API_KEY"]
    url = f"https://www.googleapis.com/youtube/v3/{endpoint}?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def main():
    env()
    sources = defaultdict(set)   # cid -> {"seed", "hashtag", "search", "extra"}
    hits = Counter()             # cid -> number of scraped Minecraft videos seen

    for cid in json.load(open(os.path.join(D, "seed_mcmap.json"))).values():
        sources[cid].add("seed")

    for path in glob.glob(os.path.join(D, "hashtag_*.txt")) + glob.glob(os.path.join(D, "search_*.txt")):
        tag = "hashtag" if "hashtag_" in path else "search"
        for line in open(path):
            cid = line.split("|", 1)[0].strip()
            if cid.startswith("UC"):
                sources[cid].add(tag)
                hits[cid] += 1

    extra = os.path.join(D, "extra_handles.txt")
    if os.path.exists(extra):
        for h in open(extra):
            h = h.strip()
            if not h:
                continue
            items = yt("channels", part="id", forHandle=h).get("items", [])
            if items:
                sources[items[0]["id"]].add("extra")
            else:
                print("no channel for", h, file=sys.stderr)

    ids = list(sources)
    out = []
    for i in range(0, len(ids), 50):   # 1 quota unit per 50 channels
        for it in yt("channels", part="snippet,statistics,contentDetails", id=",".join(ids[i:i + 50]),
                     maxResults=50).get("items", []):
            st = it["statistics"]
            out.append({
                "id": it["id"],
                "name": it["snippet"]["title"],
                "handle": it["snippet"].get("customUrl", ""),
                "subs": int(st.get("subscriberCount", 0)),
                "videos": int(st.get("videoCount", 0)),
                "uploads": it["contentDetails"]["relatedPlaylists"]["uploads"],
                "found_in": sorted(sources[it["id"]]),
                "mc_hits": hits[it["id"]],
            })
    out.sort(key=lambda c: -c["subs"])
    json.dump(out, open(os.path.join(D, "candidates.json"), "w"), indent=1, ensure_ascii=False)
    print(len(ids), "ids ->", len(out), "channels resolved")


if __name__ == "__main__":
    main()
