"""Room-type counts from printed labels (stoa_qa.room_labels) vs the fused detector, on CubiCasa val.

OCR lines: eval_out/ocr_cubicasa_val.json (modal_jobs/cubicasa_cloud.py::ocr_folder). Detector boxes: the fused
whole-image boxes from the tiling probe (eval_out/yolo/<run>/probe_tiles.json). Truth: CubiCasa annotation labels
(eval_out/yolo_data/labels/val; 98-100% = Stoa's answers). "labels" = printed-label count when the plan prints room names,
else the detector; exact-count % per question.

    venv/Scripts/python _analysis/room_labels_eval.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from stoa_qa import room_labels as R  # noqa: E402
from stoa_qa.cubicasa_ensemble import fuse  # noqa: E402
from stoa_qa.cubicasa_rules import CLASSES, QUESTION_CLASSES, select_boxes  # noqa: E402

QS = ["bedrooms", "kitchens", "bathrooms", "wc_rooms", "saunas", "balconies"]


def main() -> None:
    cfg = json.loads((REPO / "stoa_qa/cubicasa_picks.json").read_text())
    ocr = json.loads((REPO / "eval_out/ocr_cubicasa_val.json").read_text(encoding="utf-8"))
    pm = json.loads((REPO / "eval_out/yolo/m13-1280/probe_tiles.json").read_text())["yolo/images/val"]["full"]
    ps = json.loads((REPO / "eval_out/yolo/local-y11s-13c/probe_tiles.json").read_text())["yolo/images/val"]["full"]
    lbl = REPO / "eval_out/yolo_data/labels/val"
    stems = sorted(ocr)
    labelled = sum(R.count(ocr[s], "bedrooms") is not None for s in stems)
    print(f"{len(stems)} val plans; {labelled} print room names (>= {R.MIN_LABELLED})")
    res = {q: Counter() for q in QS}
    misses = {q: [] for q in QS}
    for s in stems:
        ids = [int(l.split()[0]) for l in (lbl / f"{s}.txt").read_text().split("\n") if l.strip()]
        boxes = fuse(pm[s], ps[s])
        for q in QS:
            gold = sum(CLASSES[i] in QUESTION_CLASSES[q] for i in ids)
            det = len(select_boxes(boxes, q, cfg["yolo_thresholds"][q], cfg["yolo_iou"][q]))
            c = R.count(ocr[s], q)
            lab = det if c is None else len(c["labels"])
            res[q]["det"] += det == gold
            res[q]["labels"] += lab == gold
            n_l = 0 if c is None else len(c["labels"])
            res[q]["rescue"] += (det if det > 0 or n_l == 0 else n_l) == gold   # labels only when the detector saw none
            res[q]["max"] += max(det, n_l) == gold
            if c is not None:
                res[q]["n_lab"] += 1
                res[q]["det_on_lab"] += det == gold
                res[q]["lab_on_lab"] += lab == gold
                if lab != gold and len(misses[q]) < 4:
                    misses[q].append((s, gold, lab, [ln["text"] for ln in c["labels"]]))
    for q in QS:
        r = res[q]
        n = r["n_lab"] or 1
        print(f"{q:10} all plans: detector {100 * r['det'] / len(stems):5.1f}%  labels-first {100 * r['labels'] / len(stems):5.1f}%"
              f"  rescue {100 * r['rescue'] / len(stems):5.1f}%  max {100 * r['max'] / len(stems):5.1f}%"
              f"   | on {r['n_lab']} labelled plans: detector {100 * r['det_on_lab'] / n:5.1f}%  labels {100 * r['lab_on_lab'] / n:5.1f}%")
    print("\nsome label misses (plan, truth, labels, lines):")
    for q in QS:
        for m in misses[q][:2]:
            print(" ", q, m)


if __name__ == "__main__":
    main()
