"""Quick fixed-choice decisions via OpenAI's Decisions API (gpt-6-luna, ~$0.10 / 1M input tokens, no output cost).

  decide(text, [choice("kind", "What is this?", {"mc": "Minecraft", "game": "another game"})])
  -> {"kind": ("mc", 0.97)}        predicate questions -> {"name": probability}
"""
import json, os, sys, time, urllib.error, urllib.request

URL = "https://api.openai.com/v1/decisions"


def choice(name, instructions, choices):
    return {"type": "choice", "name": name, "instructions": instructions,
            "choices": [{"value": v, "description": d} for v, d in choices.items()]}


def predicate(name, instructions):
    return {"type": "predicate", "name": name, "instructions": instructions}


def decide(text, questions):
    body = json.dumps({"model": "gpt-6-luna", "input": text, "questions": questions}).encode()
    for attempt in range(5):
        req = urllib.request.Request(URL, data=body, headers={
            "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                answers = json.load(r)["answers"]
            return {a["name"]: (a["choice"], a["confidence"]) if a["type"] == "choice" else a.get("probability", a.get("score"))
                    for a in answers}
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if attempt == 4:
                raise
            print("decide retry:", getattr(e, "code", e), file=sys.stderr)
            time.sleep(2 ** attempt)
