"""GPU half of the CubiCasa detector: train YOLO on the Volume built by cubicasa_detector.py (extract + labels), then predict.

Modal only runs GPU functions once the account has a payment method.

    modal run modal_jobs/cubicasa_train.py::fit --epochs 40
    modal run modal_jobs/cubicasa_train.py::predict --split val
    modal run modal_jobs/cubicasa_train.py::predict --split test
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
VOL = "/vol"
DATA = f"{VOL}/yolo"
RUNS = f"{VOL}/runs"
CLASSES = ["door", "window", "toilet", "sink", "shower", "bathtub", "floor",
           "sauna", "kitchen", "bedroom", "wc_room", "bathroom", "balcony"]  # same order as stoa_qa/cubicasa_rules.py

vol = modal.Volume.from_name("stoa-cubicasa", create_if_missing=True)
app = modal.App("stoa-cubicasa-detector")
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics>=8.3", "pillow")
)


@app.function(image=gpu_image, volumes={VOL: vol}, gpu="H100", cpu=16, memory=65536, timeout=8 * 3600)
def train(model: str = "yolo11s.pt", epochs: int = 40, imgsz: int = 1280, batch: int = 32, name: str = "y11s-1280") -> dict:
    from ultralytics import YOLO

    m = YOLO(model)
    m.add_callback("on_fit_epoch_end", lambda trainer: vol.commit())  # keep progress if the container dies
    m.train(data=f"{DATA}/data.yaml", epochs=epochs, imgsz=imgsz, batch=batch, workers=14, project=RUNS, name=name,
            exist_ok=True, flipud=0.5, fliplr=0.5, plots=False, patience=15)
    vol.commit()
    res = m.val(data=f"{DATA}/data.yaml", split="test", imgsz=imgsz, batch=batch, plots=False)
    return {"run": f"{RUNS}/{name}", "test_map50": float(res.box.map50), "test_map50_95": float(res.box.map),
            "per_class_map50": {CLASSES[i]: float(v) for i, v in enumerate(res.box.ap50)}}


@app.function(image=gpu_image, volumes={VOL: vol}, gpu="L40S", timeout=3600)
def predict_split(split: str = "test", name: str = "y11s-1280", imgsz: int = 1280, conf: float = 0.05) -> dict:
    """Low-confidence boxes for every plan in a split; thresholds are tuned offline on val."""
    import glob
    import os

    from ultralytics import YOLO

    m = YOLO(f"{RUNS}/{name}/weights/best.pt")
    files = sorted(glob.glob(f"{DATA}/images/{split}/*.png"))
    out = {}
    for i in range(0, len(files), 16):
        for f, r in zip(files[i:i + 16], m.predict(files[i:i + 16], imgsz=imgsz, conf=conf, verbose=False)):
            b = r.boxes
            out[os.path.basename(f)[:-4]] = [[int(c), round(float(s), 4), *[round(float(v), 1) for v in xyxy]]
                                             for c, s, xyxy in zip(b.cls, b.conf, b.xyxy)]
    return out


@app.local_entrypoint()
def fit(epochs: int = 40, model: str = "yolo11s.pt", imgsz: int = 1280, batch: int = 32, name: str = "y11s-1280"):
    print(json.dumps(train.remote(model, epochs, imgsz, batch, name), indent=1))


@app.local_entrypoint()
def predict(split: str = "test", name: str = "y11s-1280"):
    out = predict_split.remote(split, name)
    dest = REPO / "eval_out/yolo" / name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{split}.json").write_text(json.dumps(out))
    print(f"{len(out)} plans -> {dest / (split + '.json')}")
