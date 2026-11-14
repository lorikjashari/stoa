"""How well do the plan detectors count on a drawing style they were not trained on? Swiss held-out sheets.

Truth: modal_jobs/swiss_labels.py boxes (from the DXF) of the Swiss `test` split (by site; never trained on).
Each run predicts at conf 0.05 (cached in eval_out/yolo/<run>/swiss_test.json); counts use the app's val-tuned
thresholds and overlap cuts from stoa_qa/cubicasa_picks.json, exactly as an uploaded plan would be counted.

    venv/Scripts/python _analysis/swiss_detector_eval.py m13-1280:1280 local-y11s-13c:1024 mixed-s13:1024 \
        --fuse m13-1280+local-y11s-13c --fuse m13-1280+mixed-s13
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

DEVICE = "cpu" if "--cpu" in sys.argv else 0
MIXED = REPO / "eval_out" / "yolo_data_mixed"
QUESTIONS = ["doors", "windows", "toilets", "sinks", "baths"]  # objects the DXF labels completely


def weights(run: str) -> Path:
    for p in (REPO / "third_party/weights" / f"{run}.pt", REPO / "eval_out/yolo_runs" / run / "weights/best.pt"):
        if p.is_file():
            return p
    raise FileNotFoundError(run)


def predict(run: str, imgsz: int, files: list[Path], split: str) -> dict:
    out_f = REPO / "eval_out" / "yolo" / run / f"swiss_{split}.json"
    if out_f.exists():
        return json.loads(out_f.read_text())
    from ultralytics import YOLO

    m = YOLO(str(weights(run)))
    out = {}
    for i in range(0, len(files), 4):
        batch = [str(f) for f in files[i:i + 4]]
        for f, r in zip(batch, m.predict(batch, imgsz=imgsz, conf=0.05, verbose=False, device=DEVICE)):
            b = r.boxes
            out[Path(f).stem] = [[int(c), round(float(s), 4), *[round(float(v), 1) for v in xyxy]]
                                 for c, s, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]
    out_f.parent.mkdir(parents=True, exist_ok=True)
    out_f.write_text(json.dumps(out))
    return out


def truth(stem: str, split: str) -> dict:
    ids = [int(l.split()[0]) for l in (MIXED / "labels" / split / f"{stem}.txt").read_text().split("\n") if l.strip()]
    return {q: sum(CLASSES[i] in QUESTION_CLASSES[q] for i in ids) for q in QUESTIONS}


def score(name: str, boxes: dict, gold: dict, cfg: dict) -> dict:
    row = {}
    for q in QUESTIONS:
        exact = err = 0
        for stem, g in gold.items():
            n = len(select_boxes(boxes.get(stem, []), q, cfg["yolo_thresholds"].get(q, 0.5), cfg["yolo_iou"].get(q, 1.0)))
            exact += n == g[q]
            err += abs(n - g[q]) / max(1, g[q])
        row[q] = (round(100 * exact / len(gold), 1), round(100 * err / len(gold), 1))
    row["mean_exact"] = round(sum(v[0] for v in row.values()) / len(QUESTIONS), 1)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run:imgsz")
    ap.add_argument("--fuse", action="append", default=[], help="runA+runB (fused as the app fuses)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    cfg = json.loads((REPO / "stoa_qa/cubicasa_picks.json").read_text())
    files = sorted((MIXED / "images" / a.split).glob("swiss-*.png"))
    gold = {f.stem: truth(f.stem, a.split) for f in files}
    print(f"{len(gold)} Swiss {a.split} sheets; truth totals:", {q: sum(g[q] for g in gold.values()) for q in QUESTIONS})
    preds = {}
    for spec in a.runs:
        run, size = spec.split(":")
        preds[run] = predict(run, int(size), files, a.split)
    for f in a.fuse:
        x, y = f.split("+")
        preds[f] = {s: fuse(preds[x].get(s, []), preds[y].get(s, [])) for s in gold}
    print("per question: (exact-count %, mean relative error %)")
    for name, boxes in preds.items():
        print(f"{name:32}", json.dumps(score(name, boxes, gold, cfg)))


if __name__ == "__main__":
    main()
