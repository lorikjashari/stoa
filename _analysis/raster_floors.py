"""Can side-by-side floor plates be counted from room layout? Val split only."""
import json
from pathlib import Path

from stoa_qa.cubicasa_gemini import load_truth, split_members

REPO = Path(__file__).resolve().parent.parent
truth = load_truth()
preds = {}
for f in (REPO / "third_party/cubicasa_preds").glob("*.json"):
    r = json.loads(f.read_text())
    if "rooms" in r:
        preds[r["path"]] = r
val = sorted(p for p in split_members("val") if p in truth and "floors" in truth[p] and p in preds)
print("val", len(val))


def boxes(rooms):
    out = []
    for r in rooms:
        if r["class"] in ("Outdoor",):
            continue
        pts = r["poly"]
        if len(pts) < 3:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        out.append((min(xs), min(ys), max(xs), max(ys)))
    return out


def plates(rooms, gap_frac: float) -> int:
    b = boxes(rooms)
    if len(b) < 2:
        return 1

    def splits(axis):
        spans = sorted((r[axis], r[axis + 2]) for r in b)
        merged = [list(spans[0])]
        for a, c in spans[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], c)
            else:
                merged.append([a, c])
        if len(merged) == 1:
            return 1
        width = merged[-1][1] - merged[0][0]
        gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
        # a real second plate is separated by a gap, not by a doorway
        cuts = sum(g > gap_frac * width for g in gaps)
        return cuts + 1

    return max(splits(0), splits(1))


for gap in (0.04, 0.08, 0.12, 0.18, 0.25):
    ok = n = 0
    by = {}
    for p in val:
        pred = plates(preds[p]["rooms"], gap)
        g = truth[p]["floors"]
        n += 1
        ok += pred == g
        by.setdefault(g, [0, 0])
        by[g][1] += 1
        by[g][0] += pred == g
    detail = " ".join(f"{k}:{a}/{b}" for k, (a, b) in sorted(by.items()))
    print(f"gap {gap:.2f}  {ok}/{n} = {ok / n:.1%}   {detail}")

always = sum(truth[p]["floors"] == 1 for p in val)
print(f"always 1: {always}/{len(val)} = {always / len(val):.1%}")
