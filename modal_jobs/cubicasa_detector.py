"""CubiCasa5K annotations -> detector dataset, on Modal (CPU only).

Stoa's cubicasa5k images are CubiCasa5K's F1_original.png files and Stoa's count answers are derived
from each plan's model.svg. A detector trained on those annotations therefore learns exactly what
Stoa counts, and every counted item comes with a box a human can check.

Two stages, so the label rules can be tuned locally against Stoa's answers without re-parsing SVGs:

    modal run modal_jobs/cubicasa_detector.py::extract    # zip -> images + every annotated object (boxes)
    uv run python -m stoa_qa.cubicasa_rules               # fit/verify label rules against questions.jsonl
    modal run modal_jobs/cubicasa_detector.py::labels     # write YOLO labels from the rules
    modal run modal_jobs/cubicasa_train.py::fit           # train (GPU, or --device cpu)

Honesty rule: train on CubiCasa's train split, pick thresholds on val, report on the held-out test
split only. Data licence: CubiCasa5K is CC BY-NC 4.0 (non-commercial, credit Kalervo et al. 2019).
"""
from __future__ import annotations

import gzip
import json
import math
import re
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
VOL = "/vol"
ZIP = f"{VOL}/cubicasa5k.zip"
DATA = f"{VOL}/yolo"
OBJECTS = f"{VOL}/objects.json.gz"
ZIP_URL = "https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1"

vol = modal.Volume.from_name("stoa-cubicasa", create_if_missing=True)
app = modal.App("stoa-cubicasa-data")  # CPU only; training lives in cubicasa_train.py
prep_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl")
    .pip_install("svgelements", "pillow")
    .add_local_python_source("stoa_qa", ignore=["**/__pycache__/**"])
)

# ------------------------------------------------------------------ model.svg -> objects with boxes

SVG_NS = "{http://www.w3.org/2000/svg}"
TARGET_PREFIX = ("Door ", "Window ", "FixedFurniture ", "Space ")


def _mul(m, n):
    a, b, c, d, e, f = m
    a2, b2, c2, d2, e2, f2 = n
    return (a * a2 + c * b2, b * a2 + d * b2, a * c2 + c * d2, b * c2 + d * d2, a * e2 + c * f2 + e, b * e2 + d * f2 + f)


def _transform(t: str | None):
    m = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    for name, args in re.findall(r"(matrix|translate|scale|rotate)\s*\(([^)]*)\)", t or ""):
        v = [float(x) for x in re.split(r"[\s,]+", args.strip()) if x]
        if name == "matrix" and len(v) == 6:
            n = tuple(v)
        elif name == "translate" and v:
            n = (1.0, 0.0, 0.0, 1.0, v[0], v[1] if len(v) > 1 else 0.0)
        elif name == "scale" and v:
            n = (v[0], 0.0, 0.0, v[1] if len(v) > 1 else v[0], 0.0, 0.0)
        elif name == "rotate" and v:
            c, s = math.cos(math.radians(v[0])), math.sin(math.radians(v[0]))
            n = (c, s, -s, c, 0.0, 0.0)
            if len(v) == 3:
                n = _mul((1.0, 0.0, 0.0, 1.0, v[1], v[2]), _mul(n, (1.0, 0.0, 0.0, 1.0, -v[1], -v[2])))
        else:
            continue
        m = _mul(m, n)
    return m


def _floats(s: str | None) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?", s or "")]


def _local_points(el) -> list[tuple[float, float]]:
    """Points that bound one SVG shape, in its own coordinate system."""
    tag = el.tag.replace(SVG_NS, "")
    g = lambda k: float(el.get(k) or 0)  # noqa: E731
    if tag in ("polygon", "polyline"):
        v = _floats(el.get("points"))
        return list(zip(v[0::2], v[1::2]))
    if tag == "rect":
        x, y, w, h = g("x"), g("y"), g("width"), g("height")
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    if tag == "circle":
        cx, cy, r = g("cx"), g("cy"), g("r")
        return [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
    if tag == "ellipse":
        cx, cy, rx, ry = g("cx"), g("cy"), g("rx"), g("ry")
        return [(cx - rx, cy - ry), (cx + rx, cy - ry), (cx + rx, cy + ry), (cx - rx, cy + ry)]
    if tag == "line":
        return [(g("x1"), g("y1")), (g("x2"), g("y2"))]
    if tag == "path":
        try:
            import svgelements as se

            bb = se.Path(el.get("d") or "").bbox()
            if bb is not None:
                return [(bb[0], bb[1]), (bb[2], bb[1]), (bb[2], bb[3]), (bb[0], bb[3])]
        except Exception:
            pass
        v = _floats(el.get("d"))
        return list(zip(v[0::2], v[1::2]))
    return []


def _subtree_points(el, m, out, skip_text=True):
    for ch in el:
        tag = ch.tag.replace(SVG_NS, "")
        if tag in ("text", "desc", "defs", "marker"):
            continue
        mm = _mul(m, _transform(ch.get("transform")))
        if tag == "g":
            _subtree_points(ch, mm, out)
        else:
            a, b, c, d, e, f = mm
            out.extend((a * x + c * y + e, b * x + d * y + f) for x, y in _local_points(ch))


def _bbox(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)] if pts else None


def svg_objects(svg_text: str) -> dict:
    """Every annotated door / window / fixture / room with its box (model.svg coordinates) and its floor.

    One record per XML group, so counts equal the number of annotated objects by construction.
    Hidden extra floors (display:none) are included: they are drawn in the image."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(svg_text)
    objs, floors = [], []

    def walk(el, m, floor):
        for ch in el:
            if ch.tag != SVG_NS + "g":
                continue
            cls = (ch.get("class") or "").strip()
            mm = _mul(m, _transform(ch.get("transform")))
            fl = floor
            if cls.startswith("Floorplan"):
                floors.append({"name": cls, "pts": []})
                fl = len(floors) - 1
            if cls.startswith(TARGET_PREFIX):
                pts: list = []
                if cls.startswith("Space "):
                    a, b, c, d, e, f = mm
                    for poly in ch.findall(SVG_NS + "polygon"):
                        pts.extend((a * x + c * y + e, b * x + d * y + f) for x, y in _local_points(poly))
                    name = ch.find(f".//{SVG_NS}g[@id='NameLabel']/{SVG_NS}text")
                    text = (name.text or "").strip() if name is not None else ""
                else:
                    bp = [g for g in ch.findall(SVG_NS + "g") if (g.get("class") or "") == "BoundaryPolygon"]
                    if cls.startswith("FixedFurniture ") and bp:
                        _subtree_points(bp[0], _mul(mm, _transform(bp[0].get("transform"))), pts)
                    else:
                        _subtree_points(ch, mm, pts)
                    text = ""
                bb = _bbox(pts)
                if bb is not None:
                    objs.append({"c": cls, "t": text, "b": [round(v, 2) for v in bb], "f": fl})
                    if fl is not None and (cls.startswith("Space ") or cls.startswith("Door") or cls.startswith("Window")):
                        floors[fl]["pts"] += [(bb[0], bb[1]), (bb[2], bb[3])]
            walk(ch, mm, fl)

    walk(root, _transform(root.get("transform")), None)
    return {"objs": objs, "floors": [{"name": f["name"], "b": _bbox(f["pts"])} for f in floors]}


_zip = None


def _plan(args) -> dict:
    """Worker: one plan -> image in the Volume (if missing) + its objects scaled to the original image."""
    import io
    import os
    import zipfile

    from PIL import Image

    global _zip
    split, name, write_images = args  # name like 'high_quality_architectural-2003'
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
    if write_images:
        dest = f"{DATA}/images/{split}/{name}.png"
        if not os.path.exists(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fp:
                fp.write(png)
    return {"name": name, "split": split, "w": w, "h": h, **rec}


@app.function(image=prep_image, volumes={VOL: vol}, cpu=16, memory=32768, timeout=3 * 3600)
def extract_remote(write_images: bool = True) -> bytes:
    import os
    import subprocess
    import zipfile
    from multiprocessing import Pool

    if not os.path.exists(ZIP) or os.path.getsize(ZIP) < 5_000_000_000:
        print("downloading CubiCasa5K zip from Zenodo ...", flush=True)
        subprocess.run(["curl", "-sSL", "--retry", "5", "-o", ZIP, ZIP_URL], check=True)
        vol.commit()
    z = zipfile.ZipFile(ZIP)
    jobs = [(split, row.strip("/").replace("/", "-"), write_images)
            for split in ("train", "val", "test") for row in z.read(f"cubicasa5k/{split}.txt").decode().split()]
    print(f"{len(jobs)} plans", flush=True)
    results = []
    with Pool(16) as pool:
        for i, r in enumerate(pool.imap_unordered(_plan, jobs, chunksize=8)):
            results.append(r)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    blob = gzip.compress(json.dumps(results, separators=(",", ":")).encode())
    with open(OBJECTS, "wb") as f:
        f.write(blob)
    vol.commit()
    print(f"done: {sum('objs' in r for r in results)} plans, {sum('error' in r for r in results)} errors", flush=True)
    return blob


@app.function(image=prep_image, volumes={VOL: vol}, cpu=4, memory=8192, timeout=3600)
def labels_remote() -> dict:
    """Write YOLO label files for every plan from objects.json.gz using stoa_qa.cubicasa_rules."""
    import os

    from stoa_qa.cubicasa_rules import CLASSES, plan_boxes

    plans = json.loads(gzip.decompress(open(OBJECTS, "rb").read()))
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
        d = f"{DATA}/labels/{p['split']}"
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/{p['name']}.txt", "w") as f:
            f.write("\n".join(lines))
        n += 1
        boxes += len(lines)
    with open(f"{DATA}/data.yaml", "w") as f:
        f.write(f"path: {DATA}\ntrain: images/train\nval: images/val\ntest: images/test\n")
        f.write("names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    for cache in ("train", "val", "test"):  # stale ultralytics label caches would hide the new labels
        try:
            os.remove(f"{DATA}/labels/{cache}.cache")
        except FileNotFoundError:
            pass
    vol.commit()
    return {"plans": n, "boxes": boxes, "classes": CLASSES}


@app.local_entrypoint()
def extract(write_images: bool = True):
    blob = extract_remote.remote(write_images)
    dest = REPO / "eval_out/cubicasa_objects.json.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    plans = json.loads(gzip.decompress(blob))
    print(f"{sum('objs' in p for p in plans)} plans, {sum('error' in p for p in plans)} errors -> {dest}")


@app.local_entrypoint()
def labels():
    print(json.dumps(labels_remote.remote(), indent=1))


if __name__ == "__main__":  # local smoke test:  python modal_jobs/cubicasa_detector.py some/model.svg
    rec = svg_objects(open(sys.argv[1], encoding="utf-8").read())
    import collections

    print(collections.Counter(o["c"].split()[0] for o in rec["objs"]), [f["name"] for f in rec["floors"]])
