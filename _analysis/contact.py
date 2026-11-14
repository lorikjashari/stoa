import os, random, sys
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None

DATA = r"c:\Users\diar\Desktop\stoa\junction-dataset\junction-dataset\data"
src, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0
ext = (".jpg", ".png")
files = [f for f in os.listdir(os.path.join(DATA, src)) if f.lower().endswith(ext)]
random.seed(seed)
pick = random.sample(files, n)
cell = 360
cols = 4
rows = (n + cols - 1) // cols
sheet = Image.new("RGB", (cols * cell, rows * (cell + 16)), "white")
d = ImageDraw.Draw(sheet)
for i, f in enumerate(pick):
    im = Image.open(os.path.join(DATA, src, f)).convert("RGB")
    im.thumbnail((cell, cell))
    x, y = (i % cols) * cell, (i // cols) * (cell + 16)
    sheet.paste(im, (x, y + 16))
    d.text((x + 2, y + 2), f[:55], fill="red")
out = os.path.join(os.path.dirname(__file__), f"contact_{src}.jpg")
sheet.save(out, quality=80)
print(out)
