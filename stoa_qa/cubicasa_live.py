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


def predict_run(run: str, image: str | Path) -> list:
    """Boxes [cls, score, x0, y0, x1, y1] in the image's own pixels, as the offline prediction jobs write them."""
    model = _load(run)
    with _lock:  # one prediction at a time: the API serves requests from a thread pool
        r = model.predict(str(image), imgsz=_imgsz(run, model), conf=CONF, verbose=False)[0]
    b = r.boxes
    return [[int(c), round(float(s), 4), *[round(float(v), 1) for v in xyxy]]
            for c, s, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]


def _cached_run(run: str, image: Path, key: str) -> list:
    f = CACHE / key / f"{run}.json"
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
    return {"boxes": out, "runs": runs, "fused": fused, "complete": tuple(runs) == members(run), "hash": key}
