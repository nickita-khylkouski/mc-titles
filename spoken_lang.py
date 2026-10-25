"""Which language is each channel's audio in? Titles alone miss Hindi/Tagalog channels that title in English.
Asks gpt-5.4-mini with 30 titles + 6 descriptions per channel; caches to data/spoken_lang.json."""
import json, os, random, urllib.request
from concurrent.futures import ThreadPoolExecutor
from classify import env, D

PROMPT = ("You see a YouTube channel's name, 30 video titles and a few descriptions. Many Indian and Filipino "
          "creators write English titles but speak Hindi/Tagalog in the videos; look for hints like '(Hindi)', "
          "Hinglish words, Indian/Filipino names, Hindi/Tagalog descriptions. Reply JSON only: "
          '{"spoken": "<ISO 639-1 of the language spoken in the videos>", "country": "<ISO 3166 guess>", "why": "<10 words>"}')

def ask(c):
    vids = [json.loads(l) for l in open(os.path.join(D, "videos", c["id"] + ".jsonl"))]
    random.Random(0).shuffle(vids)
    msg = f"Channel: {c['name']}\nTitles:\n" + "\n".join(v["title"] for v in vids[:30]) + "\nDescriptions:\n" + \
          "\n---\n".join(v.get("desc", "")[:300] for v in vids[:6])
    body = json.dumps({"model": "gpt-5.4-mini", "response_format": {"type": "json_object"},
                       "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": msg}]}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body, headers={
        "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return c["id"], json.loads(json.load(r)["choices"][0]["message"]["content"])

env()
path = os.path.join(D, "spoken_lang.json")
cache = json.load(open(path)) if os.path.exists(path) else {}
chs = [c for c in json.load(open(os.path.join(D, "channels.json"))) if c["id"] not in cache]
with ThreadPoolExecutor(12) as ex:
    for cid, lab in ex.map(ask, chs):
        cache[cid] = lab
json.dump(cache, open(path, "w"), indent=1, ensure_ascii=False)
names = {c["id"]: c["name"] for c in json.load(open(os.path.join(D, "channels.json")))}
bad = [(names[k], v["spoken"], v["country"], v["why"]) for k, v in cache.items() if v["spoken"] != "en"]
print(len(cache), "channels;", len(bad), "not English:")
for b in sorted(bad, key=lambda b: b[1]): print("  ", *b)
