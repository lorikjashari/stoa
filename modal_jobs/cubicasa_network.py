"""Run the pretrained CubiCasa5K network (blueprint3d/cubicasa_model.py) on Modal, many plans in parallel.

On a 4-core laptop one plan takes about a minute; on Modal each plan gets its own container, so the
~400 held-out test plans finish in minutes. Results land in third_party/cubicasa_preds/<plan>.json,
the same cache blueprint3d/eval_cubicasa_model.py and the scene builder already read.

    modal run modal_jobs/cubicasa_network.py                     # held-out test split
    modal run modal_jobs/cubicasa_network.py --split val --limit 50
    uv run python -W ignore blueprint3d/eval_cubicasa_model.py --split test   # then score from the cache
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
REMOTE = "/app"

WEIGHTS_URL = "https://drive.usercontent.google.com/download?id=1gRB7ez1e4H7a9Y09lLqRuna0luZO5VRK&export=download&confirm=t"

# Model code and the 208 MB weights are fetched inside Modal's build (datacenter bandwidth),
# so nothing large is uploaded from the laptop. Only blueprint3d/*.py is added locally.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "git", "curl")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("numpy>=2.0", "opencv-python-headless>=4.10", "scipy>=1.13", "scikit-image>=0.24", "shapely>=2.0")
    .run_commands(
        f"git clone --depth 1 https://github.com/CubiCasa/CubiCasa5k.git {REMOTE}/third_party/CubiCasa5k",
        f"mkdir -p {REMOTE}/third_party/weights",
        f"curl -sL -o {REMOTE}/third_party/weights/model_best_val_loss_var.pkl '{WEIGHTS_URL}'",
        f"test $(stat -c %s {REMOTE}/third_party/weights/model_best_val_loss_var.pkl) -gt 200000000",
    )
    .add_local_dir(REPO / "blueprint3d", f"{REMOTE}/blueprint3d", ignore=["viewer", "__pycache__"])
)

app = modal.App("stoa-cubicasa-network", image=image)


@app.function(cpu=4, memory=6144, timeout=1200, max_containers=60)
def detect_one(path: str, png: bytes) -> dict:
    import os

    full = Path(REMOTE) / "junction-dataset/junction-dataset" / path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(png)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    sys.path.insert(0, f"{REMOTE}/blueprint3d")
    import eval_cubicasa_model as E  # teammate's detect(): predict + vectorize + counts

    return E.safe_detect(path)


@app.local_entrypoint()
def main(split: str = "test", limit: int = 0):
    sys.path.insert(0, str(REPO / "blueprint3d"))
    import eval_cubicasa_model as E  # stdlib-only at import time

    root = Path(E.ROOT)
    plans = sorted({json.loads(l)["path"] for l in open(root / "questions.jsonl", encoding="utf-8")
                    if "/cubicasa5k/" in l} & E.split_members(split))
    if limit:
        plans = plans[:limit]
    cache = Path(E.CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    todo = [p for p in plans if not (cache / (Path(p).stem + ".json")).exists()]
    print(f"{split}: {len(plans)} plans, {len(plans) - len(todo)} cached, {len(todo)} to run on Modal", flush=True)
    if not todo:
        return  # starmap over an empty input never returns

    done = errors = 0
    args = ((p, (root / p).read_bytes()) for p in todo)
    for res in detect_one.starmap(args, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception) or "error" in res:
            errors += 1
            print("  error:", str(res)[:200] if isinstance(res, Exception) else (res["path"], res["error"]), flush=True)
            continue
        (cache / (Path(res["path"]).stem + ".json")).write_text(json.dumps(res, separators=(",", ":")))
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(todo)} done", flush=True)
    print(f"finished: {done} new predictions, {errors} errors -> {cache}")
