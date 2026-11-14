"""Contact sheet of the sampled archive scans (5 x 3 grid, numbered) for hand labelling."""
import json, os, sys
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "junction-dataset", "junction-dataset", "data")
NAME = sys.argv[2] if len(sys.argv) > 2 else "archive_sample.json"
S = json.load(open(os.path.join(HERE, NAME)))
out = sys.argv[1]
T = 420
for src, items in S.items():
    sheet = Image.new("RGB", (5 * T, 3 * T), "white")
    d = ImageDraw.Draw(sheet)
    for k, it in enumerate(items):
        im = Image.open(os.path.join(ROOT, src, it["file"])).convert("RGB")
        im.thumbnail((T - 10, T - 10))
        x, y = (k % 5) * T, (k // 5) * T
        sheet.paste(im, (x + 5, y + 5))
        d.rectangle([x, y, x + 34, y + 26], fill="red"); d.text((x + 6, y + 6), str(k), fill="white")
    sheet.save(os.path.join(out, f"contact_{src}_{NAME[:-5]}.jpg"), quality=80)
