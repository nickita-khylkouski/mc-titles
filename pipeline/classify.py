"""Step 1b: decide which candidates are actually Minecraft YouTubers.

For each candidate with >= MIN_SUBS, grab its 25 newest upload titles (1 quota unit) and ask an
OpenAI model what share of the channel is Minecraft. Keeps share >= 50 (the video's rule),
ranks by subs, writes the top MAX_CHANNELS to data/channels.json for fetch_videos.py.
Results are cached in data/classified.json so reruns are free.
"""
import json, os, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
MIN_SUBS = int(os.environ.get("MIN_SUBS", 300_000))
MAX_CHANNELS = int(os.environ.get("MAX_CHANNELS", 600))
MODEL = "gpt-5.4-mini"
LANG = os.environ.get("LANG_ONLY", "en")  # "" keeps every language

PROMPT = """You label YouTube channels. Given a channel name and its 25 newest video titles, estimate what
percent of the channel's content is Minecraft (gameplay, mods, servers like Hypixel, Minecraft animations,
Minecraft Shorts, Minecraft roleplay all count). Also give the main language of the titles.
Reply with JSON only: {"minecraft_pct": <0-100>, "lang": "<ISO 639-1>", "what": "<5-10 word description>"}"""


def env():
    for line in open(os.path.join(ROOT, ".env")):
        k, _, v = line.strip().partition("=")
        if k:
            os.environ.setdefault(k, v)


def recent_titles(uploads):
    q = urllib.parse.urlencode({"part": "snippet", "maxResults": 25, "playlistId": uploads,
                                "key": os.environ["YOUTUBE_API_KEY"]})
    try:
        with urllib.request.urlopen("https://www.googleapis.com/youtube/v3/playlistItems?" + q, timeout=30) as r:
            return [it["snippet"]["title"] for it in json.load(r).get("items", [])]
    except urllib.error.HTTPError:
        return []


def ask(name, titles):
    body = json.dumps({
        "model": MODEL,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": PROMPT},
                     {"role": "user", "content": f"Channel: {name}\nTitles:\n" + "\n".join(titles)}],
    }).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body, headers={
        "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(json.load(r)["choices"][0]["message"]["content"])


def label(c):
    titles = recent_titles(c["uploads"])
    if not titles:
        return c["id"], {"minecraft_pct": 0, "lang": "", "what": "no public uploads"}
    return c["id"], ask(c["name"], titles)


def main():
    env()
    cands = [c for c in json.load(open(os.path.join(D, "candidates.json"))) if c["subs"] >= MIN_SUBS]
    cache_path = os.path.join(D, "classified.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [c for c in cands if c["id"] not in cache]
    print(f"{len(cands)} candidates >= {MIN_SUBS:,} subs, {len(todo)} to label")
    with ThreadPoolExecutor(4) as ex:
        for i, (cid, lab) in enumerate(ex.map(label, todo), 1):
            cache[cid] = lab
            if i % 25 == 0:
                json.dump(cache, open(cache_path, "w"), indent=1, ensure_ascii=False)
                print(f"  labelled {i}/{len(todo)}", flush=True)
    json.dump(cache, open(cache_path, "w"), indent=1, ensure_ascii=False)

    # Hand-picked names (seed list + extra handles) always stay: the 25-newest-titles check misses
    # channels that were Minecraft for years and moved on (DanTDM, Jelly, TommyInnit...).
    if LANG:
        cands = [c for c in cands if cache[c["id"]].get("lang") == LANG]
        # titles can be English while the videos are in Hindi/Tagalog/...: spoken_lang.py checks the audio language
        sp_path = os.path.join(D, "spoken_lang.json")
        spoken = json.load(open(sp_path)) if os.path.exists(sp_path) else {}
        cands = [c for c in cands if spoken.get(c["id"], {}).get("spoken", LANG) == LANG]
    forced = [c for c in cands if {"seed", "extra"} & set(c["found_in"])]
    passed = [c for c in cands if c not in forced and cache[c["id"]].get("minecraft_pct", 0) >= 50]
    passed = passed[:max(0, MAX_CHANNELS - len(forced))]
    keep = sorted(({**c, **cache[c["id"]]} for c in forced + passed), key=lambda c: -c["subs"])
    json.dump(keep, open(os.path.join(D, "channels.json"), "w"), indent=1, ensure_ascii=False)
    print(f"kept {len(keep)} Minecraft channels -> data/channels.json")


if __name__ == "__main__":
    main()
