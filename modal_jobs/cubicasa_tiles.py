"""High-resolution tiled detector for doors and windows on CubiCasa raster plans (local GPU).

Doors and windows are the most numerous raster questions and the smallest objects: at the 1024 px the
13-class detector sees, a window is often 15-30 px long. Here every plan is scaled so its long side is
2048 px and cut into overlapping 640 px tiles, so the same window is 2x larger for the detector. Boxes from
all tiles are mapped back onto the full plan and merged where tiles overlap.

    .venv/Scripts/python modal_jobs/cubicasa_tiles.py make                     # tiles + labels for train and val
    .venv/Scripts/python modal_jobs/cubicasa_tiles.py train --name tiles-dw --epochs 12
    .venv/Scripts/python modal_jobs/cubicasa_tiles.py predict --name tiles-dw  # val + test -> eval_out/yolo/tiles-dw/
    .venv/Scripts/python -m stoa_qa.cubicasa_hybrid --yolo-run local-y11s-13c --tile-run tiles-dw

Labels: stoa_qa.cubicasa_rules (door and window counts equal Stoa's answers on 100% of train plans).
Honesty rule as everywhere: train on CubiCasa's train split, tune on val, report on the held-out test split.
Needs `modal_jobs/cubicasa_local.py extract` first (images in eval_out/yolo_data/images, objects.json.gz).
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from stoa_qa.cubicasa_rules import CLASSES as FULL_CLASSES  # noqa: E402
from stoa_qa.cubicasa_rules import plan_boxes  # noqa: E402

IMAGES = REPO / "eval_out" / "yolo_data" / "images"   # full plans, written by cubicasa_local.py extract
OBJECTS = REPO / "eval_out" / "cubicasa_objects.json.gz"
DATA = REPO / "eval_out" / "tiles_dw"
RUNS = REPO / "eval_out" / "yolo_runs"
TILE_CLASSES = ["door", "window"]   # tile model ids; mapped back to FULL_CLASSES ids on output
LONG_SIDE = 2048                    # plans are rescaled to this long side before tiling
MAX_UPSCALE = 3.0
TILE = 640
OVERLAP = 200                       # larger than most doors/windows at 2048 px, so each one is whole in some tile
MIN_VISIBLE = 0.4                   # a box cut by the tile edge is labelled if this much of it is inside
EMPTY_KEEP = 0.15                   # fraction of object-free tiles kept as negatives
EDGE = 4                            # px: a predicted box this close to an inner tile edge is a cut-off fragment


def scale_of(w: int, h: int) -> float:
    return min(MAX_UPSCALE, LONG_SIDE / max(w, h))


def starts(length: int) -> list[int]:
    if length <= TILE:
        return [0]
    step = TILE - OVERLAP
    xs = list(range(0, length - TILE + 1, step))
    if xs[-1] != length - TILE:
        xs.append(length - TILE)
    return xs


def load_scaled(path: Path):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path).convert("RGB")
    s = scale_of(*im.size)
    if abs(s - 1.0) > 1e-3:
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC)
    return im, s


def tiles_of(im):
    """(x, y, tile image) covering the whole scaled plan; tiles at the right/bottom edge are padded white."""
    from PIL import Image

    for y in starts(im.height):
        for x in starts(im.width):
            t = im.crop((x, y, x + TILE, y + TILE))
            if t.size != (TILE, TILE):  # crop beyond the image pads black; repaint as white paper
                pad = Image.new("RGB", (TILE, TILE), (255, 255, 255))
                pad.paste(im.crop((x, y, min(x + TILE, im.width), min(y + TILE, im.height))), (0, 0))
                t = pad
            yield x, y, t


def _make_plan(args) -> tuple[int, int]:
    split, plan, seed = args
    src = IMAGES / split / f"{plan['name']}.png"
    if not src.exists():
        return 0, 0
    im, s = load_scaled(src)
    boxes = [(TILE_CLASSES.index(c), [v * s for v in b]) for c, b in plan_boxes(plan) if c in TILE_CLASSES]
    rnd = random.Random(seed)
    n_tiles = n_boxes = 0
    for x, y, t in tiles_of(im):
        lines = []
        for cid, (x0, y0, x1, y1) in boxes:
            cx0, cy0, cx1, cy1 = max(x0, x), max(y0, y), min(x1, x + TILE), min(y1, y + TILE)
            if cx1 - cx0 < 2 or cy1 - cy0 < 2:
                continue
            area = max(1e-6, (x1 - x0) * (y1 - y0))
            if (cx1 - cx0) * (cy1 - cy0) < MIN_VISIBLE * area:
                continue
            lines.append(f"{cid} {((cx0 + cx1) / 2 - x) / TILE:.6f} {((cy0 + cy1) / 2 - y) / TILE:.6f} "
                         f"{(cx1 - cx0) / TILE:.6f} {(cy1 - cy0) / TILE:.6f}")
        if not lines and rnd.random() > EMPTY_KEEP:
            continue
        stem = f"{plan['name']}__{x}_{y}"
        t.save(DATA / "images" / split / f"{stem}.png", optimize=False, compress_level=1)
        (DATA / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        n_tiles += 1
        n_boxes += len(lines)
    return n_tiles, n_boxes


def make(workers: int) -> None:
    plans = [p for p in json.loads(gzip.decompress(OBJECTS.read_bytes())) if "objs" in p and p["split"] in ("train", "val")]
    for split in ("train", "val"):
        (DATA / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA / "labels" / split).mkdir(parents=True, exist_ok=True)
        (DATA / "labels" / f"{split}.cache").unlink(missing_ok=True)
    jobs = [(p["split"], p, i) for i, p in enumerate(plans)]
    tot_t = tot_b = 0
    with Pool(workers) as pool:
        for i, (t, b) in enumerate(pool.imap_unordered(_make_plan, jobs, chunksize=4)):
            tot_t, tot_b = tot_t + t, tot_b + b
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(jobs)} plans, {tot_t} tiles, {tot_b} boxes", flush=True)
    (DATA / "data.yaml").write_text(f"path: {DATA.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
                                    + "".join(f"  {i}: {c}\n" for i, c in enumerate(TILE_CLASSES)))
    print(json.dumps({"plans": len(jobs), "tiles": tot_t, "boxes": tot_b, "classes": TILE_CLASSES}), flush=True)


def train(a) -> None:
    from ultralytics import YOLO

    m = YOLO(a.model)
    # symbols, not text: flips and rotations are safe here
    m.train(data=str(DATA / "data.yaml"), epochs=a.epochs, imgsz=TILE, batch=a.batch, workers=a.workers, device=0,
            project=str(RUNS), name=a.name, exist_ok=True, fliplr=0.5, flipud=0.5, degrees=0.0, plots=False,
            patience=a.patience, cache=False, close_mosaic=3)


def _overlap_frac(a, b) -> float:
    """Fraction of box a's area inside box b."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return w * h / area if area > 0 else 0.0


def merge(boxes: list) -> list:
    """Same-class boxes from overlapping tiles: keep the higher score; drop a box mostly inside a kept one
    (a door cut by one tile's edge is a smaller box inside the whole door seen by the neighbouring tile)."""
    kept: list = []
    for b in sorted(boxes, key=lambda b: -b[1]):
        if all(k[0] != b[0] or max(_overlap_frac(b[2:], k[2:]), _overlap_frac(k[2:], b[2:])) < 0.6 for k in kept):
            kept.append(b)
    return kept


def predict(a) -> None:
    from ultralytics import YOLO

    m = YOLO(str(RUNS / a.name / "weights" / "best.pt"))
    ids = [FULL_CLASSES.index(c) for c in TILE_CLASSES]
    out_dir = REPO / "eval_out" / "yolo" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        files = sorted((IMAGES / split).glob("*.png"))
        out = {}
        for i, f in enumerate(files):
            im, s = load_scaled(f)
            tiles = list(tiles_of(im))
            boxes = []
            for j in range(0, len(tiles), 16):
                chunk = tiles[j:j + 16]
                for (x, y, _), r in zip(chunk, m.predict([t for _, _, t in chunk], imgsz=TILE, conf=a.conf, verbose=False, device=0)):
                    for c, sc, xyxy in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy):
                        x0, y0, x1, y1 = (float(v) for v in xyxy)
                        # a box touching an inner tile edge is a fragment: the overlapping neighbour tile sees it whole
                        if (x0 < EDGE and x > 0) or (y0 < EDGE and y > 0) or \
                                (x1 > TILE - EDGE and x + TILE < im.width) or (y1 > TILE - EDGE and y + TILE < im.height):
                            continue
                        boxes.append([ids[int(c)], round(float(sc), 4), round((x0 + x) / s, 1), round((y0 + y) / s, 1),
                                      round((x1 + x) / s, 1), round((y1 + y) / s, 1)])
            out[f.stem] = merge(boxes)
            if (i + 1) % 100 == 0:
                print(f"  {split} {i + 1}/{len(files)}", flush=True)
        (out_dir / f"{split}.json").write_text(json.dumps(out))
        print(f"{len(out)} plans -> {out_dir / (split + '.json')}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--workers", type=int, default=4)
    t = sub.add_parser("train")
    t.add_argument("--model", default=str(REPO / "yolo11s.pt"))
    t.add_argument("--name", default="tiles-dw")
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--workers", type=int, default=6)
    t.add_argument("--patience", type=int, default=5)
    p = sub.add_parser("predict")
    p.add_argument("--name", default="tiles-dw")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    a = ap.parse_args()
    {"make": lambda: make(a.workers), "train": lambda: train(a), "predict": lambda: predict(a)}[a.cmd]()


if __name__ == "__main__":
    main()
