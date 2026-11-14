"""Draw find_views boxes on sampled sheets: archive_views.py <out.jpg> <sample json> src:k ..."""
import sys, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "blueprint3d"))
import archive_plan as A
from PIL import Image, ImageDraw
S = json.load(open(os.path.join(HERE, sys.argv[2])))
tiles = []
for a in sys.argv[3:]:
    src, k = a.split(":")
    p = os.path.join(HERE, "..", "junction-dataset", "junction-dataset", "data", src, S[src][int(k)]["file"])
    rgb = A.load_rgb(p); V = A.find_views(rgb)
    im = Image.fromarray(rgb); d = ImageDraw.Draw(im)
    for i, v in enumerate(V):
        d.rectangle(v["box"], outline=["red", "blue", "green", "magenta", "orange", "cyan", "purple", "brown"][i % 8], width=max(4, rgb.shape[1] // 300))
    print(a, len(V), [(v["box"], v["parent"]) for v in V])
    im.thumbnail((600, 600)); tiles.append(im)
sheet = Image.new("RGB", (1800, 600 * ((len(tiles) + 2) // 3)), "white")
for i, t in enumerate(tiles): sheet.paste(t, ((i % 3) * 600, (i // 3) * 600))
sheet.save(sys.argv[1])
