"""Why are windows undercounted? Annotated windows vs detector boxes on the VAL split (test stays held out).

For every annotated window (CubiCasa model.svg, via stoa_qa.cubicasa_rules) find the best detector box at the tuned
threshold and classify the miss: tiny, merged (one detection covers several annotated windows), on another floor of the
sheet, overlapping a door, or plain missed. Prints the profile and a few example plans.

    venv/Scripts/python.exe _analysis/raster_windows.py [run]
"""
import collections
import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from stoa_qa.cubicasa_gemini import load_truth  # noqa: E402
from stoa_qa.cubicasa_rules import CLASSES, plan_boxes, select_boxes  # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "m13-1280"
picks = json.loads((REPO / "stoa_qa/cubicasa_picks.json").read_text())
thr = picks["yolo_thresholds"]["windows"]
boxes = json.loads((REPO / "eval_out/yolo" / run / "val.json").read_text())
plans = {p["name"]: p for p in json.loads(gzip.decompress((REPO / "eval_out/cubicasa_objects.json.gz").read_bytes()))
         if "objs" in p and p["split"] == "val"}
truth = load_truth()
W = CLASSES.index("window")


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / u if u > 0 else 0.0


def inside(a, b):  # fraction of a inside b
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return w * h / area if area > 0 else 0.0


kinds = collections.Counter(); per_plan = []; sizes_hit, sizes_miss = [], []
gt_vs_label = collections.Counter(); low_conf = 0
for name, plan in plans.items():
    path = f"data/cubicasa5k/{name}.png"
    if "windows" not in truth.get(path, {}) or name not in boxes:
        continue
    gt = [b for c, b in plan_boxes(plan) if c == "window"]
    gt_vs_label[len(gt) == truth[path]["windows"]] += 1
    det = [b[2:6] for b in select_boxes(boxes[name], "windows", thr)]
    low = [b[2:6] for b in boxes[name] if b[0] == W and b[1] < thr]
    long_side = max(plan["w"], plan["h"])
    used = collections.Counter()
    misses = []
    for g in gt:
        best = max(range(len(det)), key=lambda i: iou(g, det[i]), default=None)
        if best is not None and iou(g, det[best]) >= 0.3:
            used[best] += 1
            sizes_hit.append(max(g[2] - g[0], g[3] - g[1]) / long_side * 1280)
            continue
        size = max(g[2] - g[0], g[3] - g[1]) / long_side * 1280  # long side at the 1280 px the detector sees
        sizes_miss.append(size)
        cover = [i for i, d in enumerate(det) if inside(g, d) > 0.6]
        if cover:
            k = "merged into a bigger detection"
        elif any(iou(g, l) >= 0.3 for l in low):
            k = "detected below the threshold"
            low_conf += 1
        elif any(o["c"].startswith("Door") and o.get("b") and iou(g, o["b"]) > 0.2 for o in plan["objs"]):
            k = "overlaps a door"
        elif size < 18:
            k = "tiny (<18 px)"
        else:
            k = "missed"
        misses.append(k)
        kinds[k] += 1
    split_extra = sum(n - 1 for n in used.values() if n > 1)
    kinds["(one detection matched several windows)"] += split_extra
    per_plan.append((len(det) - truth[path]["windows"], name, len(gt), len(det), collections.Counter(misses)))

print(f"run {run}, threshold {thr}; annotation count == Stoa answer on {gt_vs_label[True]}/{sum(gt_vs_label.values())} val plans")
print(f"annotated windows matched: {len(sizes_hit)}, missed: {len(sizes_miss)}")
for k, v in kinds.most_common():
    print(f"  {v:5}  {k}")
qs = lambda xs: [round(sorted(xs)[int(len(xs) * q)]) for q in (0.1, 0.5, 0.9)] if xs else []
print("window long side at 1280 px (p10/p50/p90): hit", qs(sizes_hit), " missed", qs(sizes_miss))
per_plan.sort()
print("most undercounted val plans:")
for d, name, g, n, m in per_plan[:8]:
    print(f"  {name:36} truth-ish {g:3} detected {n:3}  diff {d:+3}  {dict(m)}")
