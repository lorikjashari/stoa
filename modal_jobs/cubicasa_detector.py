"""Train a door / window / fixture detector on CubiCasa5K annotations, on Modal.

Stoa's cubicasa5k images are CubiCasa5K's F1_original.png files, and Stoa's count answers equal the
number of annotated objects in each plan's model.svg (checked: doors, windows, toilets, sinks,
showers+bathtubs). So a detector trained on those annotations learns exactly what Stoa counts, and
every counted item comes with a box a human can check.

Honesty rule: train on CubiCasa's train split, pick thresholds on val, report on the held-out test
split only (the same 387 plans used for the network / Gemini / Gemma comparisons).

    modal run modal_jobs/cubicasa_detector.py::prep             # CPU: zip -> YOLO dataset in a Volume, checked vs Stoa
    modal run modal_jobs/cubicasa_train.py::fit --epochs 40     # GPU (needs a payment method on the Modal account)
    modal run modal_jobs/cubicasa_train.py::predict --split val
    modal run modal_jobs/cubicasa_train.py::predict --split test
    uv run python -m stoa_qa.cubicasa_hybrid                    # tune on val, score on test

Data licence: CubiCasa5K is CC BY-NC 4.0 (non-commercial, credit Kalervo et al. 2019).
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
VOL = "/vol"
ZIP = f"{VOL}/cubicasa5k.zip"
DATA = f"{VOL}/yolo"
RUNS = f"{VOL}/runs"
ZIP_URL = "https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1"
CLASSES = ["door", "window", "toilet", "sink", "shower", "bathtub"]

vol = modal.Volume.from_name("stoa-cubicasa", create_if_missing=True)
app = modal.App("stoa-cubicasa-data")  # CPU only; GPU training lives in cubicasa_train.py

prep_image = modal.Image.debian_slim(python_version="3.12").apt_install("curl").pip_install("svgelements", "pillow")


# ------------------------------------------------------------------ labels from model.svg

def svg_label(cls: str) -> str | None:
    """CubiCasa SVG group class -> detector class (None = not a target)."""
    p = cls.split()
    if len(p) < 2:
        return None
    if p[0] == "Door":
        return None if p[1] == "None" else "door"  # 'None' = open doorway without a leaf: Stoa does not count it
    if p[0] == "Window":
        return "window"
    if p[0] == "FixedFurniture":
        k = p[1]
        if k == "Toilet":
            return "toilet"
        if "Sink" in k:  # Sink, DoubleSink, DoubleSinkRight, RoundSink, CornerSink ...: one object each
            return "sink"
        if k == "Shower":
            return "shower"
        if k.startswith("Bathtub"):
            return "bathtub"
    return None


def svg_boxes(svg_text: str) -> list[tuple[str, float, float, float, float]]:
    """All target objects as (class, x0, y0, x1, y1) in model.svg coordinates (= F1_scaled pixels).

    Multi-storey plans hide extra floors with display:none, so hidden elements are parsed too;
    svgelements then yields some groups twice, hence the de-duplication on (class, box)."""
    import io

    import svgelements as se

    svg = se.SVG.parse(io.StringIO(svg_text), reify=True, parse_display_none=True)
    out, seen = [], set()
    for g in svg.elements():
        if not isinstance(g, se.Group):
            continue
        lab = svg_label(g.values.get("class", "") or "")
        if lab is None:
            continue
        bb = g.bbox()
        if bb is None:
            continue
        key = (lab,) + tuple(round(v, 1) for v in bb)
        if key in seen:
            continue
        seen.add(key)
        out.append((lab, *bb))
    return out


_zip = None


def _plan_to_yolo(args) -> dict:
    """Worker: one plan -> image + YOLO label file. Returns its object counts (for checking vs Stoa)."""
    import io
    import os
    import zipfile

    from PIL import Image

    global _zip
    split, name = args  # name like 'high_quality_architectural-2003'
    style, num = name.rsplit("-", 1)
    base = f"cubicasa5k/{style}/{num}"
    try:
        if _zip is None:
            _zip = zipfile.ZipFile(ZIP)
        png = _zip.read(f"{base}/F1_original.png")
        w, h = Image.open(io.BytesIO(png)).size
        sw, sh = Image.open(io.BytesIO(_zip.read(f"{base}/F1_scaled.png"))).size
        boxes = svg_boxes(_zip.read(f"{base}/model.svg").decode("utf-8", "replace"))
    except Exception as e:  # missing member or unparsable svg: skip the plan, report it
        return {"name": name, "split": split, "error": repr(e)[:200]}
    fx, fy = w / sw, h / sh
    lines, counts = [], {c: 0 for c in CLASSES}
    for lab, x0, y0, x1, y1 in boxes:
        x0, x1 = max(0.0, x0 * fx), min(float(w), x1 * fx)
        y0, y1 = max(0.0, y0 * fy), min(float(h), y1 * fy)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        counts[lab] += 1
        lines.append(f"{CLASSES.index(lab)} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    os.makedirs(f"{DATA}/images/{split}", exist_ok=True)
    os.makedirs(f"{DATA}/labels/{split}", exist_ok=True)
    with open(f"{DATA}/images/{split}/{name}.png", "wb") as f:
        f.write(png)
    with open(f"{DATA}/labels/{split}/{name}.txt", "w") as f:
        f.write("\n".join(lines))
    return {"name": name, "split": split, "counts": counts}


@app.function(image=prep_image, volumes={VOL: vol}, cpu=16, memory=32768, timeout=3 * 3600)
def prepare(force: bool = False) -> dict:
    import os
    import subprocess
    import zipfile
    from multiprocessing import Pool

    if force or not os.path.exists(ZIP) or os.path.getsize(ZIP) < 5_000_000_000:
        print("downloading CubiCasa5K zip from Zenodo ...", flush=True)
        subprocess.run(["curl", "-sSL", "--retry", "5", "-o", ZIP, ZIP_URL], check=True)
        vol.commit()
    print(f"zip: {os.path.getsize(ZIP) / 1e9:.2f} GB", flush=True)
    z = zipfile.ZipFile(ZIP)
    jobs = []
    for split in ("train", "val", "test"):
        for row in z.read(f"cubicasa5k/{split}.txt").decode().split():
            jobs.append((split, row.strip("/").replace("/", "-")))
    print(f"{len(jobs)} plans to convert", flush=True)
    results = []
    with Pool(16) as pool:
        for i, r in enumerate(pool.imap_unordered(_plan_to_yolo, jobs, chunksize=8)):
            results.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    with open(f"{DATA}/data.yaml", "w") as f:
        f.write(f"path: {DATA}\ntrain: images/train\nval: images/val\ntest: images/test\n")
        f.write("names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    counts = {r["name"]: r["counts"] for r in results if "counts" in r}
    with open(f"{DATA}/counts.json", "w") as f:
        json.dump(counts, f)
    vol.commit()
    errors = [r for r in results if "error" in r]
    print(f"done: {len(counts)} plans, {len(errors)} errors", flush=True)
    return {"counts": counts, "errors": errors[:20]}


QUESTION_CLASSES = {"doors": ["door"], "windows": ["window"], "toilets": ["toilet"], "sinks": ["sink"],
                    "baths": ["shower", "bathtub"]}


@app.local_entrypoint()
def prep(force: bool = False):
    """Build the dataset, then check label counts against Stoa's answers for every plan they cover."""
    import sys

    sys.path.insert(0, str(REPO))
    from stoa_qa.cubicasa_gemini import load_truth

    res = prepare.remote(force)
    counts = res["counts"]
    dest = REPO / "eval_out/yolo"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "label_counts.json").write_text(json.dumps(counts))
    truth = load_truth()
    print(f"{len(counts)} plans labelled, {len(res['errors'])} errors {res['errors'][:3]}")
    for q, cls in QUESTION_CLASSES.items():
        n = ok = 0
        for name, c in counts.items():
            gt = truth.get(f"data/cubicasa5k/{name}.png", {}).get(q)
            if gt is not None:
                n += 1
                ok += sum(c[k] for k in cls) == gt
        print(f"  labels vs Stoa answers, {q:8}: {ok}/{n} = {ok / max(n, 1):.1%}")
