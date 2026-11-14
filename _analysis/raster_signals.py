"""Which room/icon signals explain the val misses."""
import collections
import json
from pathlib import Path

from shapely.geometry import Polygon

from stoa_qa.cubicasa_gemini import load_truth, split_members

REPO = Path(__file__).resolve().parent.parent
truth = load_truth()
preds = {}
for f in (REPO / "third_party/cubicasa_preds").glob("*.json"):
    r = json.loads(f.read_text())
    if "counts" in r and "rooms" in r:
        preds[r["path"]] = r
val = sorted(p for p in split_members("val") if p in truth and p in preds)


def poly_of(pts):
    if len(pts) < 3:
        return None
    g = Polygon(pts)
    if not g.is_valid:
        g = g.buffer(0)
    return g if not g.is_empty and g.area > 0 else None


def box(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return max(xs) - min(xs), max(ys) - min(ys)


print("val", len(val))

# bedroom residual vs other room classes
print("\nbedroom residual (pred - truth) correlation with other room counts")
classes = ["Living Room", "Undefined", "Entry", "Storage", "Outdoor", "Kitchen", "Bath"]
for cls in classes:
    # mean of that class count when under / exact / over
    buckets = {"under": [], "exact": [], "over": []}
    for p in val:
        if "bedrooms" not in truth[p]:
            continue
        d = preds[p]["counts"]["bedrooms"] - truth[p]["bedrooms"]
        c = sum(1 for r in preds[p]["rooms"] if r["class"] == cls)
        key = "under" if d < 0 else "over" if d > 0 else "exact"
        buckets[key].append(c)
    def avg(xs):
        return sum(xs) / len(xs) if xs else 0
    print(f"  {cls:14} under {avg(buckets['under']):.2f} (n={len(buckets['under'])})  exact {avg(buckets['exact']):.2f}  over {avg(buckets['over']):.2f}")

# door split hypothesis: long side / median short side
print("\ndoor polygons: split wide ones")
for factor in (1.45, 1.6, 1.8, 2.0):
    ok = n = base_ok = 0
    for p in val:
        if "doors" not in truth[p]:
            continue
        doors = [i for i in preds[p]["icons"] if i["class"] == "Door" and len(i["poly"]) >= 3]
        sides = []
        for d in doors:
            w, h = box(d["poly"])
            sides.append((max(w, h), min(w, h)))
        if not sides:
            pred = 0
        else:
            med = sorted(s[1] for s in sides)[len(sides) // 2] or 1
            pred = 0
            for long, short in sides:
                pred += max(1, round(long / max(med, short, 1))) if long > factor * max(med, 1) else 1
        n += 1
        ok += pred == truth[p]["doors"]
        base_ok += preds[p]["counts"]["doors"] == truth[p]["doors"]
    print(f"  factor {factor}: {ok}/{n} = {ok/n:.1%}  (raw {base_ok/n:.1%})")

# window split
print("\nwindow polygons: split wide ones")
for factor in (1.7, 2.0, 2.5, 3.0):
    ok = n = 0
    base = 0
    for p in val:
        if "windows" not in truth[p]:
            continue
        wins = [i for i in preds[p]["icons"] if i["class"] == "Window" and len(i["poly"]) >= 3]
        sides = []
        for d in wins:
            w, h = box(d["poly"])
            sides.append(sorted((w, h)))
        if not sides:
            pred = 0
        else:
            med = sorted(s[0] for s in sides)[len(sides) // 2] or 1
            pred = sum(max(1, round(long / med)) if long > factor * med else 1 for short, long in sides)
        n += 1
        ok += pred == truth[p]["windows"]
        base += preds[p]["counts"]["windows"] == truth[p]["windows"]
    print(f"  factor {factor}: {ok}/{n} = {ok/n:.1%}  (raw {base/n:.1%})")

# sinks: rooms that should have a sink
print("\nsink undercount vs kitchen/bath rooms missing a sink icon")
miss = have = 0
for p in val:
    if "sinks" not in truth[p]:
        continue
    d = preds[p]["counts"]["sinks"] - truth[p]["sinks"]
    if d < 0:
        miss += 1
    else:
        have += 1
print("under", miss, "not under", have)

# icon classes present
icons = collections.Counter()
for p in val:
    for i in preds[p].get("icons", []):
        icons[i["class"]] += 1
print("icons", icons.most_common())
