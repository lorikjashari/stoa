"""Cloud GPU training for the CubiCasa detectors (Modal): the 13-class plan detector and the tiled door/window detector.

Same labels, same splits and same honesty rule as the local scripts (cubicasa_local.py, cubicasa_tiles.py):
train on CubiCasa's train split, select on val, report on the held-out test split. The data comes from the
Volume built by cubicasa_detector.py (extract downloads CubiCasa5K from Zenodo; labels writes the YOLO labels
from stoa_qa.cubicasa_rules). Each GPU job copies its data to the container's local disk first.

    modal run modal_jobs/cubicasa_cloud.py::gpu_check
    modal run modal_jobs/cubicasa_detector.py::extract && modal run modal_jobs/cubicasa_detector.py::labels
    modal run --detach modal_jobs/cubicasa_cloud.py::fit_full  --name m13-1280 --model yolo11m.pt --epochs 60
    modal run --detach modal_jobs/cubicasa_cloud.py::fit_tiles --name tiles-dw-m --model yolo11m.pt --epochs 30
    modal run modal_jobs/cubicasa_cloud.py::predict --name m13-1280             # -> eval_out/yolo/m13-1280/{val,test}.json
    modal run modal_jobs/cubicasa_cloud.py::predict --name tiles-dw-m --tiles   # -> eval_out/yolo/tiles-dw-m/{val,test}.json

Two drawing styles (CubiCasa + Swiss Dwellings sheets, labels from their DXFs: modal_jobs/swiss_labels.py, uploaded to
/vol/swiss with `modal volume put stoa-cubicasa eval_out/swiss_yolo swiss`), fine-tuned from the CubiCasa runs:
    modal deploy modal_jobs/cubicasa_cloud.py   # then spawn train_mixed (survives the local client, see the end of this file)
    modal run modal_jobs/cubicasa_cloud.py::predict_both --name mixed-m13 --imgsz 1280

Data licence: CubiCasa5K is CC BY-NC 4.0 (non-commercial, credit Kalervo et al. 2019).
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
VOL = "/vol"
RUNS = f"{VOL}/runs"
LOCAL = "/tmp/data"

vol = modal.Volume.from_name("stoa-cubicasa", create_if_missing=True)
app = modal.App("stoa-cubicasa-cloud")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics>=8.3", "pillow")
    .add_local_python_source("stoa_qa", ignore=["**/__pycache__/**"])
    .add_local_file(str(REPO / "modal_jobs" / "cubicasa_tiles.py"), "/root/cubicasa_tiles.py")
)


def _tiles_module(images: str, data: str):
    """modal_jobs/cubicasa_tiles.py with its paths pointed at this container."""
    import cubicasa_tiles as T

    T.IMAGES, T.DATA, T.OBJECTS, T.RUNS = Path(images), Path(data), Path(f"{VOL}/objects.json.gz"), Path(RUNS)
    return T


def _copy_full_dataset() -> str:
    import shutil

    dest = f"{LOCAL}/yolo"
    shutil.copytree(f"{VOL}/yolo", dest, dirs_exist_ok=True)
    y = Path(dest) / "data.yaml"
    y.write_text(y.read_text().replace(f"path: {VOL}/yolo", f"path: {dest}"))
    for split in ("train", "val", "test"):
        (Path(dest) / "labels" / f"{split}.cache").unlink(missing_ok=True)
    return dest


def _copy_mixed_dataset() -> str:
    """CubiCasa (/vol/yolo) + Swiss Dwellings sheets (/vol/swiss, from modal_jobs/swiss_labels.py) in one YOLO folder.

    Both keep their own splits (Swiss by site), so CubiCasa val/test and Swiss val/test stay held out."""
    import shutil

    from stoa_qa.cubicasa_rules import CLASSES

    dest = f"{LOCAL}/mixed"
    shutil.copytree(f"{VOL}/yolo/images", f"{dest}/images", dirs_exist_ok=True)
    shutil.copytree(f"{VOL}/yolo/labels", f"{dest}/labels", dirs_exist_ok=True)
    shutil.copytree(f"{VOL}/swiss/images", f"{dest}/images", dirs_exist_ok=True)
    shutil.copytree(f"{VOL}/swiss/labels", f"{dest}/labels", dirs_exist_ok=True)
    for split in ("train", "val", "test"):
        (Path(dest) / "labels" / f"{split}.cache").unlink(missing_ok=True)
    (Path(dest) / "data.yaml").write_text(
        f"path: {dest}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    return dest


@app.function(image=image, volumes={VOL: vol}, gpu="H100", cpu=16, memory=131072, timeout=10 * 3600)
def train_mixed(base: str, epochs: int, imgsz: int, batch: int, name: str, patience: int) -> dict:
    """Fine-tune a CubiCasa detector (runs/<base>/weights/best.pt) on CubiCasa + Swiss, so it sees two drawing styles."""
    from ultralytics import YOLO

    from stoa_qa.cubicasa_rules import CLASSES

    data = _copy_mixed_dataset()
    m = YOLO(f"{RUNS}/{base}/weights/best.pt")
    m.add_callback("on_fit_epoch_end", lambda trainer: vol.commit())  # weights survive a lost container
    m.train(data=f"{data}/data.yaml", epochs=epochs, imgsz=imgsz, batch=batch, workers=14, project=RUNS, name=name,
            exist_ok=True, flipud=0.5, fliplr=0.5, plots=False, patience=patience, cache="ram")
    vol.commit()
    best = YOLO(f"{RUNS}/{name}/weights/best.pt")
    res = best.val(data=f"{data}/data.yaml", split="test", imgsz=imgsz, batch=batch, plots=False)
    out = {"run": f"{RUNS}/{name}", "base": base, "test_map50": float(res.box.map50), "test_map50_95": float(res.box.map),
           "per_class_map50": {CLASSES[i]: float(v) for i, v in enumerate(res.box.ap50)}}
    with open(f"{RUNS}/{name}/test_metrics.json", "w") as f:
        json.dump(out, f, indent=1)
    vol.commit()
    return out


@app.function(image=image, volumes={VOL: vol}, gpu="H100:8", cpu=64, memory=262144, timeout=10 * 3600)
def resume_mixed_multi(name: str, batch: int = 128) -> dict:
    """Continue a mixed run from runs/<name>/weights/last.pt on 8 GPUs (DDP). Ultralytics' DDP workers don't carry
    Python callbacks, so a thread commits the volume every 3 minutes instead of once per epoch."""
    import threading
    import time

    from ultralytics import YOLO

    from stoa_qa.cubicasa_rules import CLASSES

    data = _copy_mixed_dataset()  # recreated at the path the checkpoint remembers
    stop = threading.Event()

    def saver():
        while not stop.wait(180):
            try:
                vol.commit()
            except Exception:
                pass
    threading.Thread(target=saver, daemon=True).start()
    m = YOLO(f"{RUNS}/{name}/weights/last.pt")
    m.train(resume=True, device=list(range(8)), batch=batch, workers=6, cache=False)
    stop.set()
    vol.commit()
    best = YOLO(f"{RUNS}/{name}/weights/best.pt")
    res = best.val(data=f"{data}/data.yaml", split="test", imgsz=1024, batch=32, plots=False, device=0)
    out = {"run": f"{RUNS}/{name}", "test_map50": float(res.box.map50), "test_map50_95": float(res.box.map),
           "per_class_map50": {CLASSES[i]: float(v) for i, v in enumerate(res.box.ap50)}}
    with open(f"{RUNS}/{name}/test_metrics.json", "w") as f:
        json.dump(out, f, indent=1)
    vol.commit()
    return out


@app.function(image=image, volumes={VOL: vol}, gpu="L40S", cpu=8, memory=32768, timeout=3 * 3600)
def predict_mixed(name: str, imgsz: int) -> dict:
    """Low-confidence boxes for CubiCasa val/test and Swiss val/test ({split}, swiss_{split}); thresholds are tuned locally."""
    import os

    from ultralytics import YOLO

    m = YOLO(f"{RUNS}/{name}/weights/best.pt")
    out = {}
    for key, folder in (("val", "yolo/images/val"), ("test", "yolo/images/test"),
                        ("swiss_val", "swiss/images/val"), ("swiss_test", "swiss/images/test")):
        files = sorted(str(p) for p in Path(f"{VOL}/{folder}").glob("*.png"))
        res = {}
        for i in range(0, len(files), 16):
            for f, r in zip(files[i:i + 16], m.predict(files[i:i + 16], imgsz=imgsz, conf=0.05, verbose=False)):
                b = r.boxes
                res[os.path.basename(f)[:-4]] = [[int(c), round(float(s), 4), *[round(float(v), 1) for v in xyxy]]
                                                 for c, s, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]
        out[key] = res
    return out


@app.function(image=image, volumes={VOL: vol}, gpu="L40S", cpu=8, memory=32768, timeout=4 * 3600)
def probe_tiles(name: str, imgsz: int, tiles: list[int], sets: list[str]) -> dict:
    """Whole-image vs tiled boxes (stoa_qa.cubicasa_live.predict_tiled) for each image of each set folder on the volume,
    e.g. swiss/images/test, yolo/images/val, uploads: {set: {"full": {stem: boxes}, "t<tile>": {stem: boxes}}}."""
    import os

    from ultralytics import YOLO

    from stoa_qa.cubicasa_live import predict_tiled

    m = YOLO(f"{RUNS}/{name}/weights/best.pt")
    out = {}
    for s in sets:
        res = {"full": {}, **{f"t{t}": {} for t in tiles}}
        for f in sorted(Path(f"{VOL}/{s}").glob("*.png")):
            stem = os.path.basename(str(f))[:-4]
            r = m.predict(str(f), imgsz=imgsz, conf=0.05, verbose=False)[0]
            full = [[int(c), round(float(sc), 4), *[round(float(v), 1) for v in xyxy]]
                    for c, sc, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
            res["full"][stem] = full
            for t in tiles:
                res[f"t{t}"][stem] = predict_tiled(m, str(f), imgsz, t, full=full)
        out[s] = res
    return out


ocr_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("rapidocr_onnxruntime", "opencv-python-headless", "pillow", "numpy")
    .add_local_python_source("stoa_qa", ignore=["**/__pycache__/**"])
)


@app.function(image=ocr_image, volumes={VOL: vol}, cpu=4, memory=8192, timeout=3600)
def ocr_one(path: str) -> list:
    from stoa_qa.room_labels import read

    return read(path) or []


@app.local_entrypoint()
def ocr_folder(folder: str = "yolo/images/val", out: str = "eval_out/ocr_cubicasa_val.json"):
    """Printed text lines of every plan in a volume folder (stoa_qa.room_labels.read), fanned out over CPU containers."""
    paths = sorted(str(p.path) for p in vol.listdir(folder) if p.path.endswith(".png"))
    res = dict(zip([Path(p).stem for p in paths], ocr_one.map([f"{VOL}/{p}" for p in paths])))
    (REPO / out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(f"{len(res)} plans -> {out}")


@app.function(image=image, gpu="H100", timeout=300)
def gpu_check() -> str:
    import subprocess

    return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout


@app.function(image=image, volumes={VOL: vol}, gpu="H100", cpu=16, memory=98304, timeout=10 * 3600)
def train_full(model: str, epochs: int, imgsz: int, batch: int, name: str, patience: int) -> dict:
    from ultralytics import YOLO

    from stoa_qa.cubicasa_rules import CLASSES

    data = _copy_full_dataset()
    m = YOLO(model)
    m.add_callback("on_fit_epoch_end", lambda trainer: vol.commit())  # weights survive a lost container
    m.train(data=f"{data}/data.yaml", epochs=epochs, imgsz=imgsz, batch=batch, workers=14, project=RUNS, name=name,
            exist_ok=True, flipud=0.5, fliplr=0.5, plots=False, patience=patience, cache="ram")
    vol.commit()
    res = m.val(data=f"{data}/data.yaml", split="test", imgsz=imgsz, batch=batch, plots=False)
    return {"run": f"{RUNS}/{name}", "test_map50": float(res.box.map50), "test_map50_95": float(res.box.map),
            "per_class_map50": {CLASSES[i]: float(v) for i, v in enumerate(res.box.ap50)}}


@app.function(image=image, volumes={VOL: vol}, gpu="H100", cpu=16, memory=98304, timeout=10 * 3600)
def resume_full(name: str) -> dict:
    """Continue a stopped 13-class run from runs/<name>/weights/last.pt to its planned last epoch (same data path,
    same arguments: ultralytics stores them in the checkpoint), then score the test split."""
    from ultralytics import YOLO

    from stoa_qa.cubicasa_rules import CLASSES

    data = _copy_full_dataset()  # recreated at the same path the checkpoint remembers
    m = YOLO(f"{RUNS}/{name}/weights/last.pt")
    m.add_callback("on_fit_epoch_end", lambda trainer: vol.commit())
    m.train(resume=True)
    vol.commit()
    best = YOLO(f"{RUNS}/{name}/weights/best.pt")
    res = best.val(data=f"{data}/data.yaml", split="test", imgsz=1280, batch=16, plots=False)
    out = {"run": f"{RUNS}/{name}", "test_map50": float(res.box.map50), "test_map50_95": float(res.box.map),
           "per_class_map50": {CLASSES[i]: float(v) for i, v in enumerate(res.box.ap50)}}
    with open(f"{RUNS}/{name}/test_metrics.json", "w") as f:
        json.dump(out, f, indent=1)
    vol.commit()
    return out


@app.function(image=image, volumes={VOL: vol}, gpu="H100", cpu=16, memory=65536, timeout=10 * 3600)
def train_tiles(model: str, epochs: int, batch: int, name: str, patience: int) -> dict:
    import shutil

    from ultralytics import YOLO

    shutil.copytree(f"{VOL}/yolo/images", f"{LOCAL}/images", dirs_exist_ok=True)
    T = _tiles_module(f"{LOCAL}/images", f"{LOCAL}/tiles")
    T.make(16)  # same tiles as the local run, built on this container's disk
    m = YOLO(model)
    m.add_callback("on_fit_epoch_end", lambda trainer: vol.commit())
    m.train(data=f"{LOCAL}/tiles/data.yaml", epochs=epochs, imgsz=T.TILE, batch=batch, workers=14, project=RUNS, name=name,
            exist_ok=True, fliplr=0.5, flipud=0.5, plots=False, patience=patience, close_mosaic=3)
    vol.commit()
    return {"run": f"{RUNS}/{name}"}


@app.function(image=image, volumes={VOL: vol}, gpu="L40S", cpu=8, memory=32768, timeout=3 * 3600)
def predict_remote(name: str, tiles: bool, imgsz: int, splits: list[str]) -> dict:
    """Low-confidence boxes for every plan in each split, in full-plan pixels; thresholds are tuned locally on val."""
    import argparse
    import os

    from ultralytics import YOLO

    out = {}
    if tiles:
        T = _tiles_module(f"{VOL}/yolo/images", f"{LOCAL}/tiles")
        T.REPO = Path(LOCAL)  # predict() writes eval_out/yolo/<name>/<split>.json under REPO
        T.predict(argparse.Namespace(name=name, conf=0.05, splits=splits))
        for split in splits:
            out[split] = json.loads((Path(LOCAL) / "eval_out/yolo" / name / f"{split}.json").read_text())
        return out
    m = YOLO(f"{RUNS}/{name}/weights/best.pt")
    for split in splits:
        files = sorted(str(p) for p in Path(f"{VOL}/yolo/images/{split}").glob("*.png"))
        res = {}
        for i in range(0, len(files), 16):
            for f, r in zip(files[i:i + 16], m.predict(files[i:i + 16], imgsz=imgsz, conf=0.05, verbose=False)):
                b = r.boxes
                res[os.path.basename(f)[:-4]] = [[int(c), round(float(s), 4), *[round(float(v), 1) for v in xyxy]]
                                                 for c, s, xyxy in zip(b.cls, b.conf, b.xyxy)]
        out[split] = res
    return out


@app.local_entrypoint()
def fit_full(name: str = "m13-1280", model: str = "yolo11m.pt", epochs: int = 60, imgsz: int = 1280, batch: int = 16, patience: int = 15):
    print(json.dumps(train_full.remote(model, epochs, imgsz, batch, name, patience), indent=1))


@app.local_entrypoint()
def fit_tiles(name: str = "tiles-dw-m", model: str = "yolo11m.pt", epochs: int = 30, batch: int = 64, patience: int = 8):
    print(json.dumps(train_tiles.remote(model, epochs, batch, name, patience), indent=1))


@app.local_entrypoint()
def predict(name: str = "m13-1280", tiles: bool = False, imgsz: int = 1280, splits: str = "val,test"):
    out = predict_remote.remote(name, tiles, imgsz, splits.split(","))
    dest = REPO / "eval_out/yolo" / name
    dest.mkdir(parents=True, exist_ok=True)
    for split, boxes in out.items():
        (dest / f"{split}.json").write_text(json.dumps(boxes))
        print(f"{len(boxes)} plans -> {dest / (split + '.json')}")


@app.local_entrypoint()
def fit_mixed(base: str = "m13-1280", name: str = "mixed-m13", epochs: int = 30, imgsz: int = 1280, batch: int = 16, patience: int = 10):
    print(json.dumps(train_mixed.remote(base, epochs, imgsz, batch, name, patience), indent=1))


@app.local_entrypoint()
def predict_both(name: str = "mixed-m13", imgsz: int = 1280):
    """CubiCasa val/test + Swiss val/test boxes -> eval_out/yolo/<name>/{val,test,swiss_val,swiss_test}.json"""
    out = predict_mixed.remote(name, imgsz)
    dest = REPO / "eval_out/yolo" / name
    dest.mkdir(parents=True, exist_ok=True)
    for split, boxes in out.items():
        (dest / f"{split}.json").write_text(json.dumps(boxes))
        print(f"{len(boxes)} plans -> {dest / (split + '.json')}")


@app.local_entrypoint()
def resume(name: str = "m13-1280"):
    print(json.dumps(resume_full.remote(name), indent=1))


# Survives the local process: deploy the app, then spawn the job on Modal and return at once.
#   modal deploy modal_jobs/cubicasa_cloud.py
#   python -c "import modal; print(modal.Function.from_name('stoa-cubicasa-cloud', 'resume_full').spawn('m13-1280').object_id)"
# Progress: runs/<name>/results.csv on the volume; the result also lands in runs/<name>/test_metrics.json.
