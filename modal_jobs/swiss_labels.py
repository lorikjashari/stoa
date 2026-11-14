"""YOLO labels for the Swiss Dwellings sheets: a second drawing style for the plan detector.

The CubiCasa-only detector has seen one drawing style (Finnish real-estate plans). Swiss Dwellings ships each
floor as a PNG sheet plus the DXF it was drawn from, so exact boxes come for free: doors, windows, toilets, washbasins,
showers, bathtubs and labelled rooms are read from the DXF (stoa_qa.swiss, the one DXF reader) and mapped onto the
PNG with stoa_qa.sheets.swiss_transform (fitted on the printed room numbers). Same 13 classes and box conventions as
stoa_qa.cubicasa_rules, so the two sets train one model.

    venv/Scripts/python modal_jobs/swiss_labels.py build            # labels + mixed data.yaml under eval_out/yolo_data_mixed
    venv/Scripts/python modal_jobs/swiss_labels.py check --n 4      # draw the boxes on a few sheets (eval_out/scratch/swiss_labels)

Splits are by site (a site's buildings look alike), about 80/10/10. The CubiCasa images and labels are linked in
unchanged with their own splits, so CubiCasa's val/test stay held out. Swiss questions are answered from the DXF, never
from these detectors, so training on the Swiss sheets does not touch the Swiss benchmark.
Swiss Dwellings: CC BY 4.0 (credit per file in data/swiss_dwellings/LICENSE.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CUBI = REPO / "eval_out" / "yolo_data"          # CubiCasa set from modal_jobs/cubicasa_local.py
MIXED = REPO / "eval_out" / "yolo_data_mixed"   # CubiCasa + Swiss
SPLITS = ("train", "val", "test")

# Swiss room labels (DE/FR/IT) -> detector room class; ambiguous ones (Zimmer, Pièce, Stanza) stay unlabelled
ROOMS = [
    (re.compile(r"^(wc|dusche/wc|wc/dusche)\b", re.I), "wc_room"),
    (re.compile(r"^(bad|bain|bagno|salle de bain|dusche|douche|doccia)\b", re.I), "bathroom"),
    (re.compile(r"^(küche|kueche|cuisine|cucina)\b", re.I), "kitchen"),
    (re.compile(r"^(schlafzimmer|chambre à coucher|chambre|camera da letto|camera)\b", re.I), "bedroom"),
    (re.compile(r"^(balkon|balcon|balcone)\b", re.I), "balcony"),
]
FIXTURES = {"toilets": "toilet", "basins": "sink", "showers": "shower", "bathtubs": "bathtub"}


def split_of(stem: str) -> str:
    site = stem.split("_")[0]
    h = int(hashlib.sha1(site.encode()).hexdigest(), 16) % 10
    return "val" if h == 0 else "test" if h == 1 else "train"


def room_class(label: str | None) -> str | None:
    label = (label or "").strip()
    return next((c for rx, c in ROOMS if rx.search(label)), None)


def sheet_boxes(stem: str) -> dict | None:
    """{"w", "h", "boxes": [(class, [x0, y0, x1, y1]) in PNG pixels]} for one Swiss sheet, None when it can't be mapped."""
    from PIL import Image

    from stoa_qa import sheets, swiss

    png = sheets.DATASET / "data/swiss_dwellings" / f"{stem}.png"
    t = sheets.swiss_transform(stem)
    if t is None or not png.is_file():
        return None
    s, tx, ty = t
    w, h = Image.open(png).size
    fl = swiss.load_floor(stem)

    def box(poly):
        x0, y0, x1, y1 = poly.bounds
        return [s * x0 + tx, -s * y1 + ty, s * x1 + tx, -s * y0 + ty]

    def door_box(poly):  # the DXF door is the thin opening; CubiCasa boxes cover the leaf's swing, so square it up
        x0, y0, x1, y1 = box(poly)
        cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) / 2
        return [cx - half, cy - half, cx + half, cy + half]

    out = [("door", door_box(p)) for _, p in fl.doors]
    out += [("window", box(p)) for _, p in fl.windows]
    for f in fl.fixtures:
        k = swiss.fixture_type(f)
        if k in FIXTURES:
            out.append((FIXTURES[k], box(f["poly"])))
    rooms = [r for r in fl.rooms if r.poly is not None]
    for r in rooms:
        if (c := room_class(r.label)):
            out.append((c, box(r.poly)))
    if rooms:  # the floor: every room together (CubiCasa's 'floor' box is the drawn floor outline)
        from shapely.ops import unary_union

        out.append(("floor", box(unary_union([r.poly for r in rooms]))))
    return {"w": w, "h": h, "boxes": out}


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)  # same drive: no copy
    except OSError:
        dst.write_bytes(src.read_bytes())


def _one(stem: str) -> dict:
    from stoa_qa.cubicasa_rules import CLASSES
    from stoa_qa import sheets

    try:
        r = sheet_boxes(stem)
    except Exception as e:  # unreadable DXF or failed fit: skip the sheet, report it
        return {"stem": stem, "error": repr(e)[:200]}
    if r is None:
        return {"stem": stem, "error": "no transform"}
    w, h = r["w"], r["h"]
    lines = []
    for cls, (x0, y0, x1, y1) in r["boxes"]:
        x0, x1 = max(0.0, x0), min(float(w), x1)
        y0, y1 = max(0.0, y0), min(float(h), y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        lines.append(f"{CLASSES.index(cls)} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    sp = split_of(stem)
    name = f"swiss-{stem}"
    _link(sheets.DATASET / "data/swiss_dwellings" / f"{stem}.png", MIXED / "images" / sp / f"{name}.png")
    d = MIXED / "labels" / sp
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.txt").write_text("\n".join(lines))
    return {"stem": stem, "split": sp, "boxes": len(lines)}


def build(workers: int) -> None:
    from stoa_qa import sheets
    from stoa_qa.cubicasa_rules import CLASSES

    for sp in SPLITS:  # CubiCasa, unchanged, with its own splits
        for f in (CUBI / "images" / sp).glob("*.png"):
            _link(f, MIXED / "images" / sp / f.name)
        for f in (CUBI / "labels" / sp).glob("*.txt"):
            _link(f, MIXED / "labels" / sp / f.name)
    stems = sorted(p.stem for p in (sheets.DATASET / "data/swiss_dwellings").glob("*.png"))
    with Pool(workers) as pool:
        res = pool.map(_one, stems, chunksize=8)
    ok = [r for r in res if "error" not in r]
    (MIXED / "data.yaml").write_text(
        f"path: {MIXED.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    for sp in SPLITS:
        (MIXED / "labels" / f"{sp}.cache").unlink(missing_ok=True)
    print(json.dumps({"swiss_sheets": len(stems), "labelled": len(ok),
                      "per_split": {sp: sum(r["split"] == sp for r in ok) for sp in SPLITS},
                      "boxes": sum(r["boxes"] for r in ok),
                      "errors": [r for r in res if "error" in r][:5], "n_errors": len(res) - len(ok)}))


def check(n: int) -> None:
    """Draw the label boxes on a few sheets to see that the DXF -> PNG mapping lines up."""
    from PIL import Image, ImageDraw

    from stoa_qa import sheets

    out = REPO / "eval_out" / "scratch" / "swiss_labels"
    out.mkdir(parents=True, exist_ok=True)
    colours = {"door": "red", "window": "blue", "toilet": "magenta", "sink": "cyan", "shower": "orange",
               "bathtub": "green", "floor": "grey"}
    stems = sorted(p.stem for p in (sheets.DATASET / "data/swiss_dwellings").glob("*.png"))[::97][:n]
    for stem in stems:
        r = sheet_boxes(stem)
        if r is None:
            continue
        im = Image.open(sheets.DATASET / "data/swiss_dwellings" / f"{stem}.png").convert("RGB")
        d = ImageDraw.Draw(im)
        for cls, b in r["boxes"]:
            d.rectangle(b, outline=colours.get(cls, "purple"), width=3)
        im.save(out / f"{stem}.png")
        print(out / f"{stem}.png", {c: sum(x == c for x, _ in r["boxes"]) for c in {x for x, _ in r["boxes"]}})


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--workers", type=int, default=6)
    c = sub.add_parser("check")
    c.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    build(a.workers) if a.cmd == "build" else check(a.n)


if __name__ == "__main__":
    main()
