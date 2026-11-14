"""Run the CubiCasa detector pipeline on this machine (no Modal): extract -> labels -> train -> predict.

Same steps, same label rules and same output files as modal_jobs/cubicasa_detector.py + cubicasa_train.py,
for when a local GPU is available (Modal only runs GPU jobs once the account has a payment method).

    .venv/Scripts/python modal_jobs/cubicasa_local.py extract                 # zip -> images + annotated objects (boxes)
    .venv/Scripts/python stoa_qa/cubicasa_rules.py                            # (optional) how well do label counts match Stoa's answers
    .venv/Scripts/python modal_jobs/cubicasa_local.py labels                  # YOLO label files from stoa_qa.cubicasa_rules
    .venv/Scripts/python modal_jobs/cubicasa_local.py train --name local-y11s --epochs 30
    .venv/Scripts/python modal_jobs/cubicasa_local.py predict --name local-y11s     # val + test boxes -> eval_out/yolo/<name>/
    .venv/Scripts/python -m stoa_qa.cubicasa_hybrid --yolo-run local-y11s     # pick sources on val, report on test

Needs third_party/cubicasa5k.zip (CubiCasa5K, Zenodo 2613548, 5.5 GB, CC BY-NC 4.0: non-commercial, credit
Kalervo et al. 2019). Honesty rule as in the Modal jobs: train on CubiCasa's train split only, tune thresholds on val,
report on the held-out test split.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ZIP = REPO / "third_party" / "cubicasa5k.zip"
DATA = REPO / "eval_out" / "yolo_data"       # YOLO layout: images/<split>/<plan>.png, labels/<split>/<plan>.txt
OBJECTS = REPO / "eval_out" / "cubicasa_objects.json.gz"
RUNS = REPO / "eval_out" / "yolo_runs"
SPLITS = ("train", "val", "test")

_zip: zipfile.ZipFile | None = None


def _plan(args) -> dict:
    """Worker: one plan -> its image on disk and its annotated objects scaled to the original image pixels."""
    from PIL import Image

    from cubicasa_detector import svg_objects  # the parser used by the Modal job

    global _zip
    split, name = args
    style, num = name.rsplit("-", 1)
    base = f"cubicasa5k/{style}/{num}"
    try:
        if _zip is None:
            _zip = zipfile.ZipFile(ZIP)
        png = _zip.read(f"{base}/F1_original.png")
        w, h = Image.open(io.BytesIO(png)).size
        sw, sh = Image.open(io.BytesIO(_zip.read(f"{base}/F1_scaled.png"))).size
        rec = svg_objects(_zip.read(f"{base}/model.svg").decode("utf-8", "replace"))
    except Exception as e:  # missing member or unparsable svg: skip the plan, report it
        return {"name": name, "split": split, "error": repr(e)[:200]}
    fx, fy = w / sw, h / sh

    def scale(b):
        return None if b is None else [round(b[0] * fx, 1), round(b[1] * fy, 1), round(b[2] * fx, 1), round(b[3] * fy, 1)]

    for o in rec["objs"]:
        o["b"] = scale(o["b"])
    for f in rec["floors"]:
        f["b"] = scale(f["b"])
    dest = DATA / "images" / split / f"{name}.png"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(png)
    return {"name": name, "split": split, "w": w, "h": h, **rec}


def extract(workers: int) -> None:
    if not ZIP.exists():
        raise SystemExit(f"{ZIP} not found: download https://zenodo.org/records/2613548/files/cubicasa5k.zip first")
    z = zipfile.ZipFile(ZIP)  # a truncated download fails here, before any work is done
    jobs = [(split, row.strip("/").replace("/", "-"))
            for split in SPLITS for row in z.read(f"cubicasa5k/{split}.txt").decode().split()]
    print(f"{len(jobs)} plans", flush=True)
    results = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_plan, jobs, chunksize=8)):
            results.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    OBJECTS.parent.mkdir(parents=True, exist_ok=True)
    OBJECTS.write_bytes(gzip.compress(json.dumps(results, separators=(",", ":")).encode()))
    errs = [r for r in results if "error" in r]
    print(f"done: {sum('objs' in r for r in results)} plans, {len(errs)} errors -> {OBJECTS}")
    for r in errs[:5]:
        print("  error:", r["name"], r["error"])


def labels() -> None:
    """YOLO label files for every plan from objects.json.gz, using the same rules as the Modal job."""
    from stoa_qa.cubicasa_rules import CLASSES, plan_boxes

    plans = json.loads(gzip.decompress(OBJECTS.read_bytes()))
    n = boxes = 0
    for p in plans:
        if "objs" not in p:
            continue
        w, h = p["w"], p["h"]
        lines = []
        for cls, (x0, y0, x1, y1) in plan_boxes(p):
            x0, x1 = max(0.0, x0), min(float(w), x1)
            y0, y1 = max(0.0, y0), min(float(h), y1)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            lines.append(f"{CLASSES.index(cls)} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
        d = DATA / "labels" / p["split"]
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{p['name']}.txt").write_text("\n".join(lines))
        n += 1
        boxes += len(lines)
    (DATA / "data.yaml").write_text(
        f"path: {DATA.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    for split in SPLITS:  # stale ultralytics label caches would hide the new labels
        (DATA / "labels" / f"{split}.cache").unlink(missing_ok=True)
    print(json.dumps({"plans": n, "boxes": boxes, "classes": CLASSES}))


def train(a) -> None:
    from ultralytics import YOLO

    m = YOLO(a.model)
    data = Path(a.data or DATA)
    m.train(data=str(data / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers, device=0,
            project=str(RUNS), name=a.name, exist_ok=True, resume=a.resume, flipud=0.5, fliplr=0.5, plots=False,
            patience=a.patience, cache=a.cache or False)
    res = m.val(data=str(data / "data.yaml"), split="test", imgsz=a.imgsz, batch=a.batch, plots=False, workers=a.workers)
    from stoa_qa.cubicasa_rules import CLASSES

    print(json.dumps({"run": str(RUNS / a.name), "test_map50": float(res.box.map50), "test_map50_95": float(res.box.map),
                      "per_class_map50": {CLASSES[i]: float(v) for i, v in enumerate(res.box.ap50)}}, indent=1))


def predict(a) -> None:
    """Low-confidence boxes for every plan in val and test; thresholds are tuned on val in cubicasa_hybrid."""
    from ultralytics import YOLO

    m = YOLO(str(RUNS / a.name / "weights" / "best.pt"))
    out_dir = REPO / "eval_out" / "yolo" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        files = sorted((Path(a.data or DATA) / "images" / split).glob("*.png"))
        out = {}
        for i in range(0, len(files), 8):
            batch = [str(f) for f in files[i:i + 8]]
            for f, r in zip(batch, m.predict(batch, imgsz=a.imgsz, conf=a.conf, verbose=False, device=0, augment=a.tta)):
                b = r.boxes
                out[Path(f).stem] = [[int(c), round(float(s), 4), *[round(float(v), 1) for v in xyxy]]
                                     for c, s, xyxy in zip(b.cls, b.conf, b.xyxy)]
        (out_dir / f"{split}.json").write_text(json.dumps(out))
        print(f"{len(out)} plans -> {out_dir / (split + '.json')}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--workers", type=int, default=6)
    sub.add_parser("labels")
    t = sub.add_parser("train")
    t.add_argument("--model", default="yolo11s.pt")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--imgsz", type=int, default=1280)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--workers", type=int, default=6)
    t.add_argument("--patience", type=int, default=12)
    t.add_argument("--name", default="local-y11s")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--data", default=None, help="YOLO data folder (default eval_out/yolo_data; CubiCasa + Swiss: eval_out/yolo_data_mixed)")
    t.add_argument("--cache", choices=["disk", "ram"], default=None, help="cache resized images (disk: much faster epochs)")
    p = sub.add_parser("predict")
    p.add_argument("--name", default="local-y11s")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--tta", action="store_true")
    p.add_argument("--data", default=None, help="YOLO data folder to predict (default eval_out/yolo_data)")
    a = ap.parse_args()
    {"extract": lambda: extract(a.workers), "labels": labels, "train": lambda: train(a), "predict": lambda: predict(a)}[a.cmd]()


if __name__ == "__main__":
    main()
