"""Tag every still-"auto" cluster as mc / game / irl with the Decisions API, using the sample titles in the
review file. irl -> drop. Then `map.py --apply` to write it out. MAP_SRC=thumb for the thumbnail map."""
import json, os, re
from concurrent.futures import ThreadPoolExecutor

from classify import env
from map import NAMES, REVIEW, kind_of

env()
names = json.load(open(NAMES))
samples, cur = {}, None
for line in open(REVIEW):
    m = re.match(r"## (\d+) \|", line)
    if m:
        cur = int(m.group(1))
    elif cur is not None:
        samples[cur] = line.strip().split(" / ")
todo = [o for o in names if o.get("auto") and o["id"] in samples]
with ThreadPoolExecutor(8) as ex:
    for o, kind in zip(todo, ex.map(lambda o: kind_of(samples[o["id"]]), todo)):
        o["type"], o["drop"] = ("game" if kind == "game" else "mc"), kind == "irl"
        print(f"{o['id']:4d} {kind:5s} {o['name']}")
json.dump(names, open(NAMES, "w"), ensure_ascii=False)
