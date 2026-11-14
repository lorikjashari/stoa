"""Count thresholds for a detector trained on CubiCasa + Swiss: tuned on both val splits, reported on both test splits.

Boxes: eval_out/yolo/<run>/{val,test,swiss_val,swiss_test}.json (modal_jobs/cubicasa_cloud.py::predict_both).
Truth: label files (CubiCasa: annotation rules; Swiss: DXF via modal_jobs/swiss_labels.py). Swiss truth covers only
object classes + unambiguous room labels, so room questions are tuned and scored on CubiCasa alone.
Writes the picked thresholds to eval_out/yolo/<run>/thresholds.json (the app reads them for that run).

    venv/Scripts/python _analysis/mixed_thresholds.py mixed-m13
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from stoa_qa.cubicasa_rules import CLASSES, QUESTION_CLASSES, select_boxes  # noqa: E402

OBJECTS = ["doors", "windows", "toilets", "sinks", "baths"]
ROOMS = ["floors", "saunas", "kitchens", "bedrooms", "wc_rooms", "bathrooms", "balconies"]
THRS = [round(0.05 * i, 2) for i in range(2, 17)]
IOUS = [1.0, 0.5, 0.3]
LBL = {"val": REPO / "eval_out/yolo_data/labels/val", "test": REPO / "eval_out/yolo_data/labels/test",
       "swiss_val": REPO / "eval_out/yolo_data_mixed/labels/val", "swiss_test": REPO / "eval_out/yolo_data_mixed/labels/test"}


def gold(split: str, stem: str) -> dict:
    ids = [int(l.split()[0]) for l in (LBL[split] / f"{stem}.txt").read_text().split("\n") if l.strip()]
    return {q: sum(CLASSES[i] in QUESTION_CLASSES[q] for i in ids) for q in OBJECTS + ROOMS}


def acc(boxes: dict, split: str, q: str, thr: float, iou: float) -> tuple[int, int]:
    ok = n = 0
    for stem, b in boxes.items():
        if not (LBL[split] / f"{stem}.txt").exists():
            continue
        ok += len(select_boxes(b, q, thr, iou)) == gold(split, stem)[q]
        n += 1
    return ok, n


def main() -> None:
    run = sys.argv[1]
    d = REPO / "eval_out/yolo" / run
    B = {s: json.loads((d / f"{s}.json").read_text()) for s in LBL}
    old = json.loads((REPO / "stoa_qa/cubicasa_picks.json").read_text())
    picked = {"thresholds": {}, "iou": {}}
    print(f"{'question':10} {'thr':>5} {'iou':>4} | CubiCasa test: new (old thr) | Swiss test: new (old thr)")
    tot = {"ct": [0, 0, 0], "st": [0, 0, 0]}
    for q in OBJECTS + ROOMS:
        tune = ["val", "swiss_val"] if q in OBJECTS else ["val"]
        best = max(((t, i) for t in THRS for i in IOUS),
                   key=lambda ti: sum(acc(B[s], s, q, *ti)[0] for s in tune))
        picked["thresholds"][q], picked["iou"][q] = best
        o = (old["yolo_thresholds"][q], old["yolo_iou"][q])
        ct, ct_old = acc(B["test"], "test", q, *best), acc(B["test"], "test", q, *o)
        line = f"{q:10} {best[0]:5} {best[1]:4} | {100 * ct[0] / ct[1]:5.1f}% ({100 * ct_old[0] / ct[1]:5.1f}%)"
        tot["ct"][0] += ct[0]; tot["ct"][1] += ct_old[0]; tot["ct"][2] += ct[1]
        if q in OBJECTS:
            st, st_old = acc(B["swiss_test"], "swiss_test", q, *best), acc(B["swiss_test"], "swiss_test", q, *o)
            line += f"        | {100 * st[0] / st[1]:5.1f}% ({100 * st_old[0] / st[1]:5.1f}%)"
            tot["st"][0] += st[0]; tot["st"][1] += st_old[0]; tot["st"][2] += st[1]
        print(line)
    print(f"{'ALL':21} | {100 * tot['ct'][0] / tot['ct'][2]:5.1f}% ({100 * tot['ct'][1] / tot['ct'][2]:5.1f}%)"
          f"        | {100 * tot['st'][0] / tot['st'][2]:5.1f}% ({100 * tot['st'][1] / tot['st'][2]:5.1f}%)")
    (d / "thresholds.json").write_text(json.dumps(picked, indent=1))


if __name__ == "__main__":
    main()
