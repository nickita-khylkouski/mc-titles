"""Hand-quality labels for a random sample of thumbnails, from a vision model (Gemini flash-lite), so the
"what thumbnails get views" analysis can use real looks instead of zero-shot text phrases.

The zero-shot way (Gemini text phrase · thumbnail vector) was checked by eye on 2026-10-24 and was poor:
only ~3/10 of the top-5% "shocked face" thumbnails had one; dark, empty images scored high on every phrase.
So: label N thumbnails here -> look_probe.py trains one linear probe per look on the 768-d thumbnail vectors
and checks it on held-out labels -> analyze.py uses the probes.

Writes data/thumb_labels.jsonl ({"id", look: 0/1, ...}). Resumable.
  N=4000 WORKERS=4 nice uv run python pipeline/label_thumbs.py
"""
import base64, json, os, random, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
OUT = os.path.join(D, "thumb_labels.jsonl")
MODEL = os.environ.get("MODEL", "gemini-flash-lite-latest")
N, PER, WORKERS = int(os.environ.get("N", 4000)), 8, int(os.environ.get("WORKERS", 4))
LOOKS = {
    "shocked_face": "a face (real or cartoon or Minecraft skin) with a shocked, scared or surprised open-mouth expression",
    "real_person": "a real human person or face (photo or facecam), not a drawing or game character",
    "red_marks": "a red arrow or a red circle drawn on top to point something out",
    "big_text": "large text overlaid on the image (big enough to read at a glance)",
    "no_text": "no text, numbers or letters anywhere on the image",
    "dark": "mostly dark: night, caves, black background, horror mood",
    "two_sides": "a versus / comparison layout: two characters facing off or a split image (before/after, noob vs pro)",
    "giant_thing": "one huge thing dominates: a giant build, monster, mob or structure",
    "treasure": "diamonds, gold, money or piles of loot",
    "character_focus": "one Minecraft-style character (a skin) is the clear main subject, shown big",
    "raw_gameplay": "looks like a plain, unedited gameplay screenshot (no added text, arrows, cut-outs or effects)",
    "cartoon": "drawn or 3D-animated art rather than a game screenshot or photo",
}
lock = threading.Lock()


def env():
    for line in open(os.path.join(ROOT, ".env")):
        k, _, v = line.strip().partition("=")
        if k:
            os.environ.setdefault(k, v)


def sample():
    chans = [c["id"] for c in json.load(open(os.path.join(D, "channels.json")))]
    vids = []
    for cid in chans:
        mc = json.load(open(os.path.join(D, "is_mc", cid + ".json")))
        for l, p in zip(open(os.path.join(D, "videos", cid + ".jsonl")), mc):
            v = json.loads(l)
            if p >= 0.3 and os.path.exists(os.path.join(D, "thumbs", v["id"][:2], v["id"] + ".jpg")):
                vids.append(v["id"])
    random.Random(0).shuffle(vids)
    return vids[:N]


def ask(batch):
    q = "\n".join(f"- {k}: {d}" for k, d in LOOKS.items())
    parts = [{"text": f"These are {len(batch)} YouTube thumbnails, numbered 0-{len(batch) - 1} in order. For each one, "
                      f"answer true/false for every property:\n{q}\nJudge only what is visible. Reply JSON: a list of "
                      f"{len(batch)} objects in order, each with every property as a boolean."}]
    for i, v in enumerate(batch):
        parts += [{"text": f"Thumbnail {i}:"},
                  {"inline_data": {"mime_type": "image/jpeg",
                                   "data": base64.b64encode(open(os.path.join(D, "thumbs", v[:2], v + ".jpg"), "rb").read()).decode()}}]
    item = {"type": "OBJECT", "properties": {k: {"type": "BOOLEAN"} for k in LOOKS}, "required": list(LOOKS)}
    body = json.dumps({"contents": [{"parts": parts}], "generationConfig": {
        "responseMimeType": "application/json", "responseSchema": {"type": "ARRAY", "items": item}, "temperature": 0,
        "thinkingConfig": {"thinkingLevel": os.environ.get("THINK", "minimal")}}}).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    for attempt in range(6):
        req = urllib.request.Request(url, data=body, headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            ans = json.loads(d["candidates"][0]["content"]["parts"][0]["text"])
            if len(ans) != len(batch):
                raise ValueError(f"got {len(ans)} answers for {len(batch)}")
            return [{"id": v, **{k: int(bool(a.get(k))) for k in LOOKS}} for v, a in zip(batch, ans)]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, KeyError) as e:
            if attempt == 5 or (isinstance(e, urllib.error.HTTPError) and e.code in (400, 401, 403, 404)):
                raise
            time.sleep(min(60, 2 ** attempt))


def main():
    env()
    done = {json.loads(l)["id"] for l in open(OUT)} if os.path.exists(OUT) else set()
    todo = [v for v in sample() if v not in done]
    print(f"{len(todo)} thumbnails to label with {MODEL}", flush=True)
    batches = [todo[s:s + PER] for s in range(0, len(todo), PER)]
    n = 0

    def one(b):
        nonlocal n
        try:
            rows = ask(b)
        except Exception as e:
            print("FAILED batch:", repr(e)[:200], flush=True)
            return
        with lock, open(OUT, "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
            n += len(rows)
            if n % 400 < PER:
                print(f"  {n}/{len(todo)}", flush=True)
    list(ThreadPoolExecutor(WORKERS).map(one, batches))
    print("done", flush=True)


if __name__ == "__main__":
    main()
