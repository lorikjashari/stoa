"""Pick a scale-free WC/bathroom split on val polygon areas. Does not read the test split."""
import json
from pathlib import Path

from stoa_qa.cubicasa_gemini import load_truth, split_members

REPO = Path(__file__).resolve().parent.parent
truth = load_truth()
val = split_members("val")
preds = {}
for f in (REPO / "third_party/cubicasa_preds").glob("*.json"):
    r = json.loads(f.read_text())
    if r.get("path") in val and "rooms" in r:
        preds[r["path"]] = r
print("plans", len(preds))


def area(pts):
    acc = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        acc += x0 * y1 - x1 * y0
    return abs(acc) / 2


def areas(pred):
    scale = float(pred.get("scale") or 1)
    out = []
    for r in pred["rooms"]:
        if r["class"] == "Bath" and len(r["poly"]) >= 3:
            out.append(area(r["poly"]) * scale * scale)
    return out


def score(rule):
    ok = {"wc_rooms": [0, 0], "bathrooms": [0, 0], "baths": [0, 0]}
    for p, pred in preds.items():
        a = areas(pred)
        wc, bath = rule(a)
        gold = truth[p]
        for key, pred_v in (("wc_rooms", wc), ("bathrooms", bath), ("baths", len(a))):
            if key in gold:
                ok[key][1] += 1
                ok[key][0] += pred_v == gold[key]
    return ok


rules = {
    "all bath rooms are bathrooms, no wc": lambda a: (0, len(a)),
    "area < 4000 is wc": lambda a: (sum(x < 4000 for x in a), sum(x >= 4000 for x in a)),
    "area < 8000 is wc": lambda a: (sum(x < 8000 for x in a), sum(x >= 8000 for x in a)),
    "area < 15000 is wc": lambda a: (sum(x < 15000 for x in a), sum(x >= 15000 for x in a)),
    "smaller than half the largest": lambda a: (
        sum(x < 0.5 * max(a) for x in a) if a else 0,
        sum(x >= 0.5 * max(a) for x in a) if a else 0,
    ),
    "smaller than 0.35 of the largest": lambda a: (
        sum(x < 0.35 * max(a) for x in a) if a else 0,
        sum(x >= 0.35 * max(a) for x in a) if a else 0,
    ),
}
for name, rule in rules.items():
    ok = score(rule)
    bits = []
    for k, (a, n) in ok.items():
        bits.append(f"{k} {a / n:.1%} ({n})" if n else f"{k} -")
    print(f"{name:36} {'  '.join(bits)}")
