"""Contact sheet of every jpg in a folder (labelled with file names): archive_contact_dir.py <dir> <out.jpg>"""
import os, sys
from PIL import Image, ImageDraw
d, out = sys.argv[1], sys.argv[2]
fs = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
T, C = 360, 6
sheet = Image.new("RGB", (C * T, T * ((len(fs) + C - 1) // C)), "white")
dr = ImageDraw.Draw(sheet)
for i, f in enumerate(fs):
    im = Image.open(os.path.join(d, f)).convert("RGB"); im.thumbnail((T - 8, T - 24))
    x, y = (i % C) * T, (i // C) * T
    sheet.paste(im, (x + 4, y + 20)); dr.text((x + 4, y + 4), f[:-4], fill="red")
sheet.save(out, quality=85)
