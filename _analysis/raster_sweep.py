"""Sweep component-size thresholds against Stoa's val answers. Uses whatever is in the cache so far."""
import json
from pathlib import Path

from stoa_qa.cubicasa_gemini import load_truth, split_members

REPO = Path(__file__).resolve().parent.parent
truth = load_truth()
val = split_members("val")
poly = {}
for f in (REPO / "third_party/cubicasa_preds").glob("*.json"):
    r = json.loads(f.read_text())
    if "counts" in r:
        poly[r["path"]] = r["counts"]

seg = {}
for line in (REPO / "eval_out/seg_components.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line)
        if r["path"] in val:
            seg[r["path"]] = r
print("val seg cached", len(seg))

ICON = {"doors": "Door", "windows": "Window", "toilets": "Toilet", "sinks": "Sink", "baths": "Bathtub", "saunas": "Sauna Bench"}
ROOM = {"kitchens": "Kitchen", "bedrooms": "Bed Room", "bathrooms": "Bath", "balconies": "Outdoor"}


def n_at(areas, minimum):
    return sum(a >= minimum for a in areas)


def score(key, pred_of):
    ok = n = base = 0
    for p, rec in seg.items():
        if key not in truth[p]:
            continue
        n += 1
        ok += pred_of(rec) == truth[p][key]
        if p in poly and key in poly[p]:
            base += poly[p][key] == truth[p][key]
    return ok, n, base


print(f"{'key':12} {'thr':>6} {'cc':>7} {'poly':>7} {'n':>5}")
for key, cls in {**ICON, **ROOM}.items():
    kind = "icons" if key in ICON else "rooms"
    name = ICON.get(key) or ROOM[key]
    best = None
    for thr in (5, 10, 20, 40, 80, 120, 200, 400, 800):
        ok, n, base = score(key, lambda rec, thr=thr, kind=kind, name=name: n_at(rec[kind].get(name, []), thr))
        if n and (best is None or ok > best[0]):
            best = (ok, n, base, thr)
    if best:
        ok, n, base, thr = best
        print(f"{key:12} {thr:6} {ok / n:7.1%} {base / n:7.1%} {n:5}")
