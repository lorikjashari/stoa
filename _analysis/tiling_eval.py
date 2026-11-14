"""Whole-image vs tiled detection (stoa_qa.cubicasa_live.predict_tiled), scored from the Modal probe boxes.

Input: eval_out/yolo/<run>/probe_tiles.json from modal_jobs/cubicasa_cloud.py::probe_tiles (Swiss test, CubiCasa val,
uploads). Truth: label files (Swiss: from the DXF; CubiCasa: annotation rules, 98-100% = Stoa). Counts use the app's
val-tuned thresholds. Tiling is applied only when the image's long side exceeds --min-side (smaller plans keep the
whole-image boxes), which is how the app would switch it on.

    venv/Scripts/python _analysis/tiling_eval.py --runs m13-1280 local-y11s-13c --min-side 1600
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from stoa_qa.cubicasa_ensemble import fuse  # noqa: E402
from stoa_qa.cubicasa_rules import CLASSES, QUESTION_CLASSES, select_boxes  # noqa: E402

QS = ["doors", "windows", "toilets", "sinks", "baths"]
SETS = {"swiss/images/test": REPO / "eval_out/yolo_data_mixed/labels/test",
        "yolo/images/val": REPO / "eval_out/yolo_data/labels/val"}


def truth(lbl: Path) -> dict:
    ids = [int(l.split()[0]) for l in lbl.read_text().split("\n") if l.strip()]
    return {q: sum(CLASSES[i] in QUESTION_CLASSES[q] for i in ids) for q in QS}


def counts(boxes: list, cfg: dict) -> dict:
    return {q: len(select_boxes(boxes, q, cfg["yolo_thresholds"][q], cfg["yolo_iou"][q])) for q in QS}


def size_of(set_name: str, stem: str) -> int:
    from PIL import Image

    p = {"swiss/images/test": REPO / "eval_out/yolo_data_mixed/images/test",
         "yolo/images/val": REPO / "eval_out/yolo_data/images/val",
         "uploads": REPO / "eval_out/uploads/cubicasa5k"}[set_name] / f"{stem}.png"
    return max(Image.open(p).size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs=2, default=["m13-1280", "local-y11s-13c"])
    ap.add_argument("--min-side", type=int, nargs="+", default=[0, 1600, 2000])
    a = ap.parse_args()
    cfg = json.loads((REPO / "stoa_qa/cubicasa_picks.json").read_text())
    probes = {r: json.loads((REPO / "eval_out/yolo" / r / "probe_tiles.json").read_text()) for r in a.runs}
    modes = [m for m in probes[a.runs[0]]["uploads"] if m != "full"]
    sizes = {s: {stem: size_of(s, stem) for stem in probes[a.runs[0]][s]["full"]} for s in probes[a.runs[0]]}

    def boxes(run_or_fused: str, s: str, mode: str, stem: str, min_side: int) -> list:
        m = mode if mode == "full" or sizes[s][stem] > min_side else "full"
        if run_or_fused == "fused":
            return fuse(probes[a.runs[0]][s][m][stem], probes[a.runs[1]][s][m][stem])
        return probes[run_or_fused][s][m][stem]

    for s, lbl_dir in SETS.items():
        stems = list(probes[a.runs[0]][s]["full"])
        gold = {st: truth(lbl_dir / f"{st}.txt") for st in stems}
        big = sum(sizes[s][st] > 1600 for st in stems)
        print(f"\n== {s}: {len(stems)} images ({big} with long side > 1600 px); exact-count % per question, mean")
        for who in [*a.runs, "fused"]:
            for mode in ["full", *modes]:
                for ms in (a.min_side if mode != "full" else [0]):
                    row = {q: 0 for q in QS}
                    for st in stems:
                        c = counts(boxes(who, s, mode, st, ms), cfg)
                        for q in QS:
                            row[q] += c[q] == gold[st][q]
                    pct = {q: round(100 * v / len(stems), 1) for q, v in row.items()}
                    tag = f"{who:16} {mode:6}" + (f" >{ms}px" if mode != "full" else "       ")
                    print(f"{tag:34} mean {sum(pct.values()) / len(QS):5.1f}  {pct}")
    print("\n== uploads (no truth): counts per mode, fused")
    for stem in probes[a.runs[0]]["uploads"]["full"]:
        print(stem[:8], sizes["uploads"][stem], {m: counts(boxes("fused", "uploads", m, stem, 0), cfg) for m in ["full", *modes]})


if __name__ == "__main__":
    main()
