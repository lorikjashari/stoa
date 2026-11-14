"""Deploy the whole app (API + web UI + linked 3D viewer) on Modal behind a public URL.

    modal run modal_jobs/serve.py::load_data     # once: Stoa's dataset is downloaded inside Modal into a Volume
    modal deploy modal_jobs/serve.py             # prints the public https URL; scales to zero when idle

CPU only (no payment method needed). Free-form questions and the vision fallback need the Featherless
key in the Modal secret `stoa-keys`:

    modal secret create stoa-keys FEATHERLESS_API_KEY=... --force

Precomputed raster predictions (third_party/cubicasa_preds, eval_out/gemini) are small JSON files and
are shipped with the image, so raster plans answer instantly for the plans that were evaluated.
"""
from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
DATA_URL = "https://fwjydgxpttajiqpfmqtj.supabase.co/storage/v1/object/public/stoa-junction-2026/junction-dataset.zip"

vol = modal.Volume.from_name("stoa-dataset", create_if_missing=True)
app = modal.App("plumbline")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "unzip")
    .pip_install("ezdxf>=1.3", "shapely>=2.0", "ifcopenshell>=0.8", "pymupdf>=1.24", "numpy", "pillow", "python-dotenv",
                 "fastapi", "uvicorn[standard]")
    .env({"STOA_DATASET": "/data/junction-dataset"})
    .add_local_dir(REPO / "stoa_qa", "/app/stoa_qa", ignore=["**/__pycache__/**"])
    .add_local_dir(REPO / "web", "/app/web")
    .add_local_dir(REPO / "blueprint3d", "/app/blueprint3d", ignore=["viewer/scenes/**", "**/__pycache__/**"])
    .add_local_dir(REPO / "third_party/cubicasa_preds", "/app/third_party/cubicasa_preds")
    .add_local_dir(REPO / "eval_out/gemini", "/app/eval_out/gemini")
)


@app.function(image=image, volumes={"/data": vol}, cpu=4, memory=8192, timeout=3 * 3600)
def load_data(force: bool = False):
    import os
    import subprocess

    if os.path.isfile("/data/junction-dataset/questions.jsonl") and not force:
        print("dataset already in the volume")
        return
    subprocess.run(["curl", "-sSL", "--retry", "5", "-o", "/data/dataset.zip", DATA_URL], check=True)
    subprocess.run(["unzip", "-q", "-o", "/data/dataset.zip", "-d", "/data"], check=True)
    os.remove("/data/dataset.zip")
    vol.commit()
    print("dataset ready:", os.listdir("/data/junction-dataset"))


@app.function(image=image, volumes={"/data": vol}, secrets=[modal.Secret.from_name("stoa-keys")],
              cpu=2, memory=6144, scaledown_window=900, timeout=600)
@modal.concurrent(max_inputs=24)
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, "/app")
    from stoa_qa.api import app as fastapi_app

    return fastapi_app
