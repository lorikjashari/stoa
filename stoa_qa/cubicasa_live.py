"""Run the trained YOLO11 plan detectors on an image that has no precomputed boxes (a new or uploaded plan).

The app normally reads detector boxes that were predicted offline for CubiCasa's val/test plans
(eval_out/yolo/<run>/{val,test}.json). For any other image this module runs the same detectors here, with the
same settings the offline predictions used (conf 0.05, each run's own image size), and returns boxes in the same
[cls, score, x0, y0, x1, y1] format. When both members of the picked fused run are available, their boxes are fused
with stoa_qa.cubicasa_ensemble.fuse exactly as the offline run was; the val-tuned thresholds then apply unchanged.
With only one member available its boxes are used as they are (marked `fused: False`), since the thresholds were
tuned for the fused pair the counts are then less certain.

Weights (gitignored, never committed): third_party/weights/<run>.pt or third_party/weights/<run>/best.pt, e.g.

    modal volume get stoa-cubicasa runs/m13-1280/weights/best.pt third_party/weights/m13-1280.pt

STOA_YOLO_WEIGHTS overrides the folder; STOA_NO_LIVE_YOLO=1 turns live detection off. Needs the optional extra:
uv sync --extra detect.
Boxes are cached per image content hash and run under eval_out/yolo/live/<hash>/<run>.json.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import threading
from functools import lru_cache
from pathlib import Path

from .cubicasa_ensemble import fuse

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "eval_out" / "yolo" / "live"
CONF = 0.05  # same as the offline predictions; thresholds are applied later, per question
# the image size each run was trained and predicted at (modal_jobs/cubicasa_cloud.py, cubicasa_local.py)
IMGSZ = {"m13-1280": 1280, "local-y11s-13c": 1024}
# fused runs -> the two runs they were made from, in the order given to stoa_qa.cubicasa_ensemble
ENSEMBLES = {"ens-m13-s13": ("m13-1280", "local-y11s-13c")}
# Large sheets: symbols shrink below what the detectors learned when the whole sheet is scaled to imgsz, so doors,
# windows and fixtures are detected on overlapping 960 px tiles (predict_tiled). Measured with the Modal probe
# (_analysis/tiling_eval.py): 88 held-out Swiss sheets (2480 px) fused mean exact count 28.9% -> 42.7%; CubiCasa val
# with tiling above 2000 px 78.5% -> 78.4% (one question in 2,000); 640 and 1280 px tiles scored lower.
TILE = 960
TILE_MIN_SIDE = 2000

_lock = threading.Lock()
_models: dict[str, object] = {}


def weights_dir() -> Path:
    return Path(os.environ.get("STOA_YOLO_WEIGHTS") or REPO / "third_party" / "weights")


def weight_file(run: str) -> Path | None:
    d = weights_dir()
    return next((p for p in (d / f"{run}.pt", d / run / "best.pt", d / run / "weights" / "best.pt") if p.is_file()), None)


@lru_cache(maxsize=1)
def has_ultralytics() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


def members(run: str | None) -> tuple[str, ...]:
    """The single-model runs behind a picked run (the run itself when it is not a fused one)."""
    if not run:
        return ()
    return ENSEMBLES.get(run, (run,))


def available_runs(run: str | None) -> list[str]:
    """Members of `run` that can be executed here: ultralytics installed and weights on disk."""
    if os.environ.get("STOA_NO_LIVE_YOLO") or not has_ultralytics():
        return []
    return [m for m in members(run) if weight_file(m) is not None]


def available(run: str | None) -> bool:
    return bool(available_runs(run))


def image_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:24]


def _load(run: str):
    """The YOLO model of one run, loaded once per process."""
    with _lock:
        if run not in _models:
            from ultralytics import YOLO

            from .cubicasa_rules import CLASSES

            model = YOLO(str(weight_file(run)))
            if len(model.names) != len(CLASSES):  # class ids index cubicasa_rules.CLASSES: other weights would mislabel
                raise ValueError(f"{weight_file(run)} has {len(model.names)} classes, the plan detector needs {len(CLASSES)}")
            _models[run] = model
        return _models[run]


def _imgsz(run: str, model) -> int:
    if run in IMGSZ:
        return IMGSZ[run]
    size = (getattr(model, "overrides", None) or {}).get("imgsz")  # stored in the checkpoint's train args
    return int(size[0] if isinstance(size, (list, tuple)) else size) if size else 1280


def tiled(image: str | Path) -> bool:
    """Is this sheet large enough to be detected in tiles (long side over TILE_MIN_SIDE px)?"""
    from PIL import Image

    with Image.open(image) as im:
        return max(im.size) > TILE_MIN_SIDE


def predict_run(run: str, image: str | Path) -> list:
    """Boxes [cls, score, x0, y0, x1, y1] in the image's own pixels, as the offline prediction jobs write them;
    on a large sheet doors, windows and fixtures come from tiles (see TILE)."""
    model = _load(run)
    with _lock:  # one prediction at a time: the API serves requests from a thread pool
        r = model.predict(str(image), imgsz=_imgsz(run, model), conf=CONF, verbose=False)[0]
        b = r.boxes
        full = [[int(c), round(float(s), 4), *[round(float(v), 1) for v in xyxy]]
                for c, s, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]
        if tiled(image):
            return predict_tiled(model, image, _imgsz(run, model), TILE, full=full)
    return full


OBJECT_CLASSES = range(6)  # door, window, toilet, sink, shower, bathtub (cubicasa_rules.CLASSES); rooms/floor are 6..12


def tile_grid(w: int, h: int, tile: int, overlap: float = 0.25) -> list[tuple[int, int, int, int]]:
    """Overlapping tile windows (x0, y0, x1, y1) covering a w x h image; one window when it already fits."""
    def starts(n: int) -> list[int]:
        if n <= tile:
            return [0]
        step = max(1, int(tile * (1 - overlap)))
        s = list(range(0, n - tile, step))
        if n - tile - s[-1] < tile // 10:  # the last window would nearly repeat the one before: move that one instead
            s[-1] = n - tile
            return s
        return s + [n - tile]
    return [(x, y, min(w, x + tile), min(h, y + tile)) for y in starts(h) for x in starts(w)]


def _nms(boxes: list, iou: float = 0.5) -> list:
    from .cubicasa_rules import _iou

    kept: list = []
    for b in sorted(boxes, key=lambda r: -r[1]):
        if all(k[0] != b[0] or _iou(b[2:6], k[2:6]) <= iou for k in kept):
            kept.append(b)
    return kept


def predict_tiled(model, image, imgsz: int, tile: int, conf: float = CONF, full: list | None = None) -> list:
    """Boxes for a large sheet whose symbols are too small at imgsz: doors, windows and fixtures come from overlapping
    tiles of `tile` source pixels (each scaled to imgsz), rooms and floors from the whole-image pass `full`.

    A box touching a tile edge that is not the image edge is dropped (the neighbouring tile, overlapping by 25%, sees
    it whole), then duplicates across tiles are merged per class (IoU > 0.5)."""
    from PIL import Image

    im = Image.open(image).convert("RGB")
    w, h = im.size
    if full is None:
        r = model.predict(im, imgsz=imgsz, conf=conf, verbose=False)[0]
        full = [[int(c), float(s), *map(float, xyxy)] for c, s, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
    grid = tile_grid(w, h, tile)
    if len(grid) == 1:
        return full
    objs = []
    for i in range(0, len(grid), 8):
        wins = grid[i:i + 8]
        for (x0, y0, x1, y1), r in zip(wins, model.predict([im.crop(g) for g in wins], imgsz=imgsz, conf=conf, verbose=False)):
            for c, s, (a, b, cc, d) in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                if int(c) not in OBJECT_CLASSES:
                    continue
                edge = 3
                if (a < edge and x0 > 0) or (b < edge and y0 > 0) or (cc > x1 - x0 - edge and x1 < w) or (d > y1 - y0 - edge and y1 < h):
                    continue
                objs.append([int(c), float(s), a + x0, b + y0, cc + x0, d + y0])
    out = _nms(objs) + [b for b in full if b[0] not in OBJECT_CLASSES]
    return [[b[0], round(b[1], 4), *[round(v, 1) for v in b[2:6]]] for b in out]


def _cached_run(run: str, image: Path, key: str) -> list:
    f = CACHE / key / (f"{run}.t{TILE}.json" if tiled(image) else f"{run}.json")
    if f.exists():
        return json.loads(f.read_text())
    boxes = predict_run(run, image)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(boxes))
    return boxes


def boxes(run: str | None, image: str | Path) -> dict | None:
    """Live boxes for an image: {"boxes", "runs", "fused", "complete", "hash"}; None when no member of `run` can run here.

    `complete` is True when the boxes come from exactly the run(s) the thresholds were tuned for."""
    runs = available_runs(run)
    image = Path(image)
    if not runs or not image.is_file():
        return None
    key = image_hash(image)
    per_run = [_cached_run(m, image, key) for m in runs]
    fused = len(per_run) == 2
    out = fuse(per_run[0], per_run[1]) if fused else per_run[0]
    return {"boxes": out, "runs": runs, "fused": fused, "complete": tuple(runs) == members(run), "hash": key,
            "tiled": tiled(image)}


# ------------------------------------------------------------------ uploads: the detector trained on two drawing styles
LIVE_PICKS = Path(__file__).resolve().parent / "cubicasa_live_picks.json"


@lru_cache(maxsize=1)
def live_config() -> dict | None:
    """The upload detector (cubicasa_live_picks.json): run, image size, thresholds tuned on CubiCasa + Swiss val, tile rule."""
    return json.loads(LIVE_PICKS.read_text()) if LIVE_PICKS.exists() else None


def _n_objects(boxes: list, cfg: dict) -> int:
    from .cubicasa_rules import select_boxes

    return sum(len(select_boxes(boxes, q, cfg["yolo_thresholds"][q], cfg["yolo_iou"][q]))
               for q in ("doors", "windows", "toilets", "sinks", "baths"))


def _cached(run: str, key: str, mode: str, compute) -> list:
    f = CACHE / key / f"{run}.{mode}.json"
    if f.exists():
        return json.loads(f.read_text())
    out = compute()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    return out


def boxes_live(image: str | Path) -> dict | None:
    """Boxes of the upload detector, or None when it isn't installed (then boxes() runs the dataset's pair).

    The whole sheet is detected at the run's image size. A sheet larger than a tile is also detected in tiles, and the
    tiled boxes are used only when they find far more confident objects (at least `times` as many and `plus` more):
    then the symbols were too small for the whole-sheet pass (dense dormitory sheets, small-scale plans). On Swiss and
    CubiCasa sheets this rule never tiles, where whole-sheet detection is the better one (Swiss test 64.8% vs 59.8%)."""
    cfg = live_config()
    if cfg is None or os.environ.get("STOA_NO_LIVE_YOLO") or not has_ultralytics() or weight_file(cfg["run"]) is None:
        return None
    image = Path(image)
    if not image.is_file():
        return None
    run, size, rule = cfg["run"], cfg["imgsz"], cfg["tile_rule"]
    key = image_hash(image)

    def full_pass() -> list:
        model = _load(run)
        with _lock:
            r = model.predict(str(image), imgsz=size, conf=CONF, verbose=False)[0]
        b = r.boxes
        return [[int(c), round(float(s), 4), *[round(float(v), 1) for v in xyxy]]
                for c, s, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]

    full = _cached(run, key, "full", full_pass)
    out, used_tiles = full, False
    from PIL import Image

    with Image.open(image) as im:
        big = max(im.size) > rule["tile"] * 1.25
    if big:
        def tile_pass() -> list:
            model = _load(run)
            with _lock:
                return predict_tiled(model, image, size, rule["tile"], full=full)
        tiles = _cached(run, key, f"t{rule['tile']}", tile_pass)
        nf, nt = _n_objects(full, cfg), _n_objects(tiles, cfg)
        if nt >= rule["times"] * max(nf, 1) and nt - nf >= rule["plus"]:
            out, used_tiles = tiles, True
    return {"boxes": out, "runs": [run], "fused": False, "complete": True, "hash": key, "tiled": used_tiles,
            "selected_on": cfg["selected_on"],
            "cfg": {"yolo_thresholds": cfg["yolo_thresholds"], "yolo_iou": cfg["yolo_iou"]}}
