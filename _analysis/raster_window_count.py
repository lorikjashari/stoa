"""Sweep a smarter window counter on the VAL split: two-tier threshold with a per-plan shape check, and splitting of
merged ribbon windows. Prints the best settings by exact-match rate on val; test is not read.

    venv/Scripts/python.exe _analysis/raster_window_count.py [run]
"""
import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from stoa_qa.cubicasa_gemini import load_truth, split_members  # noqa: E402
from stoa_qa.cubicasa_rules import window_count  # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "m13-1280"
q = sys.argv[2] if len(sys.argv) > 2 else "windows"
boxes = json.loads((REPO / "eval_out/yolo" / run / "val.json").read_text())
truth = load_truth()
val = [(boxes[Path(p).stem], truth[p][q]) for p in split_members("val") if p in truth and q in truth[p] and Path(p).stem in boxes]

grid = {"hi": [0.4, 0.45, 0.5, 0.55, 0.6], "lo": [0.15, 0.2, 0.25, 0.3, 0.35, 1.0], "tol": [0.25, 0.4, 0.6],
        "split": [1.6, 1.8, 2.2, 99.0]}
res = []
for hi, lo, tol, split in itertools.product(*grid.values()):
    if lo >= hi and lo != 1.0:
        continue
    ok = sum(window_count(b, q, hi, lo, tol, split) == g for b, g in val)
    res.append((ok / len(val), hi, lo, tol, split))
res.sort(reverse=True)
base = max(r for r in res if r[2] == 1.0 and r[4] == 99.0)
print(f"{q} on val ({len(val)} plans): plain threshold best {base[0]:.1%} at {base[1]}")
for r in res[:8]:
    print(f"  {r[0]:.1%}  hi {r[1]}  lo {r[2]}  thickness tol {r[3]}  split at {r[4]}x")
