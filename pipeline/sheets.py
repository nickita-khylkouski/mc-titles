"""Contact sheets of sample thumbnails per cluster, for naming clusters by eye.

Reads data/clusters_review{SFX}_ids.json (written by map.py) and the local thumbnails,
writes data/sheets{SFX}/sheet_NN.png with 4 clusters per image (6x4 thumbs each, header = id + current name).
  MAP_SRC=thumb uv run python pipeline/sheets.py          # only clusters still marked auto
  MAP_SRC=thumb uv run python pipeline/sheets.py --all
"""
import json, os, sys

from PIL import Image, ImageDraw, ImageFont

from map import D, NAMES, REVIEW, SFX

W, H, COLS, ROWS, PER = 160, 90, 8, 2, 8
HEAD = 22

names = {o["id"]: o for o in json.load(open(NAMES))}
samples = {int(k): v for k, v in json.load(open(REVIEW.replace(".txt", "_ids.json"))).items()}
todo = [k for k in sorted(samples) if "--all" in sys.argv or names.get(k, {}).get("auto")]
out = os.path.join(D, "sheets" + SFX)
os.makedirs(out, exist_ok=True)
font = ImageFont.load_default(size=15)

for n, s in enumerate(range(0, len(todo), PER)):
    block = todo[s:s + PER]
    sheet = Image.new("RGB", (COLS * W, len(block) * (HEAD + ROWS * H)), "white")
    draw = ImageDraw.Draw(sheet)
    for b, k in enumerate(block):
        y0 = b * (HEAD + ROWS * H)
        o = names.get(k, {})
        draw.text((4, y0 + 3), f"#{k}  {o.get('name', '?')}  [{o.get('type', 'mc')}{' DROP' if o.get('drop') else ''}]",
                  fill="black", font=font)
        for i, vid in enumerate(samples[k][:COLS * ROWS]):
            p = os.path.join(D, "thumbs", vid[:2], vid + ".jpg")
            if os.path.exists(p):
                sheet.paste(Image.open(p).convert("RGB").resize((W, H)), ((i % COLS) * W, y0 + HEAD + (i // COLS) * H))
    sheet.save(os.path.join(out, f"sheet_{n:02d}.png"))
print(f"{len(todo)} clusters -> {out}")
