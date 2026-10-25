"""docs/embeddings.jpg (the three maps side by side) + JPG copies of the screenshots:  uv run python docs/compose.py"""
import os
from PIL import Image, ImageDraw, ImageFont
D = os.path.dirname(os.path.abspath(__file__))
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
maps = [("embed_title.png", "Titles"), ("embed_rich.png", "Descriptions"), ("embed_thumb.png", "Thumbnails")]
S, PAD, CAP = 1200, 40, 110
out = Image.new("RGB", (3 * S + 4 * PAD, S + CAP + PAD), "black")
dr = ImageDraw.Draw(out)
f = ImageFont.truetype(FONT, 56)
for i, (p, name) in enumerate(maps):
    x = PAD + i * (S + PAD)
    out.paste(Image.open(os.path.join(D, p)).convert("RGB").resize((S, S), Image.LANCZOS), (x, CAP))
    w = dr.textlength(name, font=f)
    dr.text((x + (S - w) / 2, 30), name, font=f, fill=(235, 235, 235))
out.save(os.path.join(D, "embeddings.jpg"), quality=88)
for p in ("map.png", "search.png", "cluster.png"):
    im = Image.open(os.path.join(D, p)).convert("RGB")
    im.resize((im.width // 4 * 3, im.height // 4 * 3), Image.LANCZOS).save(os.path.join(D, p.replace(".png", ".jpg")), quality=88)
