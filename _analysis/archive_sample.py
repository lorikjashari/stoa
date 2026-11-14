"""Pick a fixed random sample of archive scans per source and print their sizes."""
import json, os, random, sys
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.join(os.path.dirname(__file__), "..", "junction-dataset", "junction-dataset", "data")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
NAME = "archive_sample.json" if SEED == 7 else f"archive_sample_{SEED}.json"
out = {}
for src in ["wikimedia", "finna", "europeana", "habs"]:
    files = sorted(f for f in os.listdir(os.path.join(ROOT, src)) if f.lower().endswith(".jpg"))
    pick = random.Random(SEED).sample(files, N)
    out[src] = []
    for f in pick:
        with Image.open(os.path.join(ROOT, src, f)) as im:
            out[src].append({"file": f, "w": im.width, "h": im.height, "mode": im.mode})
            print(src, f, im.width, im.height, im.mode)
json.dump(out, open(os.path.join(os.path.dirname(__file__), NAME), "w"), indent=1)
