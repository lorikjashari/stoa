"""Fuse two detector runs into one set of boxes (a small weighted-box fusion), written as a run the hybrid and the app use.

Boxes of the same class that the two models both found (IoU >= 0.5) become one box with the averaged score and
coordinates; a box only one model found keeps `lone` times its score, so it needs to be confident on its own to count.
The two runs differ in model size and resolution (YOLO11m 1280 px on Modal, YOLO11s 1024 px on a laptop), so their
mistakes differ. Chosen on val (mean exact-match over the 12 questions 87.2% -> 88.3%); thresholds are then tuned on
val by stoa_qa.cubicasa_hybrid as for any run.

    python -m stoa_qa.cubicasa_ensemble m13-1280 local-y11s-13c --out ens-m13-s13
    python -m stoa_qa.cubicasa_hybrid --yolo-run ens-m13-s13
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cubicasa_rules import _iou

REPO = Path(__file__).resolve().parent.parent


def fuse(a: list, b: list, lone: float = 0.5, match: float = 0.5) -> list:
    """Boxes [cls, score, x0, y0, x1, y1] of two models -> fused boxes in the same format."""
    out, used = [], set()
    for x in sorted(a, key=lambda r: -r[1]):
        cand = [i for i in range(len(b)) if i not in used and b[i][0] == x[0]]
        j = max(cand, key=lambda i: _iou(x[2:6], b[i][2:6]), default=None)
        if j is not None and _iou(x[2:6], b[j][2:6]) >= match:
            used.add(j)
            y = b[j]
            out.append([x[0], round((x[1] + y[1]) / 2, 4)] + [round((u + v) / 2, 1) for u, v in zip(x[2:6], y[2:6])])
        else:
            out.append([x[0], round(x[1] * lone, 4)] + list(x[2:6]))
    out += [[y[0], round(y[1] * lone, 4)] + list(y[2:6]) for i, y in enumerate(b) if i not in used]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lone", type=float, default=0.5)
    a = ap.parse_args()
    dest = REPO / "eval_out/yolo" / a.out
    dest.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        ra = json.loads((REPO / "eval_out/yolo" / a.run_a / f"{split}.json").read_text())
        rb = json.loads((REPO / "eval_out/yolo" / a.run_b / f"{split}.json").read_text())
        fused = {k: fuse(ra[k], rb.get(k, []), a.lone) for k in ra}
        (dest / f"{split}.json").write_text(json.dumps(fused))
        print(f"{len(fused)} plans -> {dest / (split + '.json')}")


if __name__ == "__main__":
    main()
