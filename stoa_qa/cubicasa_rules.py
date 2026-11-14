"""CubiCasa5K annotation classes -> detector classes, and a check of those rules against Stoa's answers.

Stoa's cubicasa5k answers are counts of CubiCasa's own model.svg annotations. The rules below were fitted
on CubiCasa's TRAIN split only (_analysis/raster_svg_rules.py) and reproduce Stoa's answer there for
98-100% of plans per question. A detector trained on these labels therefore learns what Stoa counts, and
every counted item comes with a box a human can check.

Stdlib only: this module is also imported inside the Modal job that writes the YOLO labels.

    modal run modal_jobs/cubicasa_detector.py::extract   # or: modal_jobs/cubicasa_local.py extract
    uv run python -m stoa_qa.cubicasa_rules              # how often do label counts equal Stoa's answers?
"""
from __future__ import annotations

# Order is the YOLO class id. The first seven are the original detector classes (keep their ids stable).
CLASSES = ["door", "window", "toilet", "sink", "shower", "bathtub", "floor",
           "sauna", "kitchen", "bedroom", "wc_room", "bathroom", "balcony"]
QUESTION_CLASSES = {"doors": ["door"], "windows": ["window"], "toilets": ["toilet"], "sinks": ["sink"],
                    "baths": ["shower", "bathtub"], "floors": ["floor"], "saunas": ["sauna"],
                    "kitchens": ["kitchen"], "bedrooms": ["bedroom"], "wc_rooms": ["wc_room"],
                    "bathrooms": ["bathroom"], "balconies": ["balcony"]}


def label_of(cls: str) -> str | None:
    """Annotation group class (e.g. 'Door Swing Beside', 'FixedFurniture DoubleSink') -> detector class."""
    p = cls.split()
    if len(p) < 2:
        return None
    if p[0] == "Door":
        return None if p[1] == "None" else "door"  # 'None' = open doorway without a leaf: Stoa does not count it
    if p[0] == "Window":
        return "window"
    if p[0] == "FixedFurniture":
        k = p[1]
        if k == "Toilet":
            return "toilet"
        if "Sink" in k:  # Sink, DoubleSink, RoundSink ...: a double-bowl sink is one object, as in Stoa's question
            return "sink"
        if k in ("Shower", "ShowerCab"):  # ShowerScreen is a glass panel, not a shower
            return "shower"
        if k.startswith("Bathtub"):
            return "bathtub"
    return None


def room_of(cls: str, text: str) -> str | None:
    """Space annotation ('Space Bath', name label 'WC') -> room detector class.

    The name label decides WC vs bathroom: Finnish plans mark separate toilets 'WC' (also 'WC/PH' rooms)."""
    if not cls.startswith("Space "):
        return None
    kind = cls[6:]
    label = " ".join((text or "").upper().split())
    if label.startswith("WC"):
        return "wc_room"
    if kind.startswith("Bath"):
        return "bathroom"
    if kind == "Bedroom":
        return "bedroom"
    if kind.startswith("Kitchen"):
        return "kitchen"
    if kind.startswith("Outdoor Balcony"):  # includes glazed balconies, not terraces or porches
        return "balcony"
    return None


def _inside(a: list[float], b: list[float], frac: float = 0.8) -> bool:
    """Is at least `frac` of box a's area inside box b?"""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return area > 0 and w * h >= frac * area


def plan_boxes(plan: dict) -> list[tuple[str, list[float]]]:
    """(class, [x0, y0, x1, y1]) for one plan record from cubicasa_detector.extract (original image pixels)."""
    objs = [o for o in plan["objs"] if o.get("b")]
    out = [(lab, o["b"]) for o in objs if (lab := label_of(o["c"]))]
    # a shower symbol drawn inside a shower cabin (or tub) is one fixture
    wet = sorted([(lab, b) for lab, b in out if lab in ("shower", "bathtub")], key=lambda x: -(x[1][2] - x[1][0]) * (x[1][3] - x[1][1]))
    kept: list[tuple[str, list[float]]] = []
    for lab, b in wet:
        if not any(_inside(b, k) for _, k in kept):
            kept.append((lab, b))
    out = [x for x in out if x[0] not in ("shower", "bathtub")] + kept
    # a sauna is its stove; a few plans annotate a Sauna space without a stove symbol
    stoves = [o["b"] for o in objs if "SaunaStove" in o["c"]]
    spaces = [o["b"] for o in objs if o["c"] == "Space Sauna"]
    out += [("sauna", b) for b in (stoves if len(stoves) >= len(spaces) else spaces)]
    out += [(lab, o["b"]) for o in objs if (lab := room_of(o["c"], o.get("t", "")))]
    out += [("floor", f["b"]) for f in plan.get("floors", []) if f.get("b")]
    return out


def _iou(a: list[float], b: list[float]) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def select_boxes(boxes: list, question: str, thr: float, iou: float = 1.0) -> list:
    """Detector boxes [cls, score, x0, y0, x1, y1] counted for one question.

    Keeps boxes of the question's classes scoring >= thr, then drops a box that overlaps an already kept,
    higher-scoring one by more than `iou` across those classes (YOLO's own NMS is per class, so a shower and a
    tub drawn on one fixture would otherwise count twice). iou=1.0 keeps everything. Both are tuned on val."""
    ids = {CLASSES.index(c) for c in QUESTION_CLASSES[question]}
    cand = sorted((b for b in boxes if b[0] in ids and b[1] >= thr), key=lambda b: -b[1])
    if iou >= 1.0:
        return cand
    kept: list = []
    for b in cand:
        if all(_iou(b[2:6], k[2:6]) <= iou for k in kept):
            kept.append(b)
    return kept


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return (xs[n // 2] + xs[(n - 1) // 2]) / 2 if n else 0.0


def opening_boxes(boxes: list, question: str, hi: float, lo: float = 1.0, tol: float = 0.4, split: float = 99.0) -> list:
    """Counted openings (windows / doors) as (box, n): n > 1 for a box that spans several merged openings.

    On val most missed windows were either detected just under the threshold or merged with a neighbour into one box.
    So: boxes >= hi count; a weaker box (lo <= score < hi) counts too when it is as thick as this plan's confident openings
    (same wall, same symbol) and does not overlap one; a box `split` times longer than the plan's typical opening counts as
    round(length / typical). lo=1.0 and split=99 turn both off. All parameters are chosen on val."""
    strong = select_boxes(boxes, question, hi)
    if len(strong) < 3:  # too few to know this plan's window style: plain threshold
        return [(b, 1) for b in strong]
    thick = _median([min(b[4] - b[2], b[5] - b[3]) for b in strong])
    length = _median([max(b[4] - b[2], b[5] - b[3]) for b in strong])
    kept = list(strong)
    for b in select_boxes(boxes, question, lo) if lo < hi else []:
        if b[1] >= hi:
            continue
        t = min(b[4] - b[2], b[5] - b[3])
        if thick > 0 and abs(t / thick - 1) <= tol and all(_iou(b[2:6], k[2:6]) < 0.3 and not _inside(b[2:6], k[2:6], 0.5) for k in kept):
            kept.append(b)
    out = []
    for b in kept:
        long = max(b[4] - b[2], b[5] - b[3])
        out.append((b, max(1, round(long / length)) if length > 0 and long > split * length else 1))
    return out


def window_count(boxes: list, question: str, hi: float, lo: float = 1.0, tol: float = 0.4, split: float = 99.0) -> int:
    return sum(n for _, n in opening_boxes(boxes, question, hi, lo, tol, split))


def main() -> None:
    import collections
    import gzip
    import json
    from pathlib import Path

    from .cubicasa_gemini import load_truth

    f = Path(__file__).resolve().parent.parent / "eval_out" / "cubicasa_objects.json.gz"
    if not f.exists():
        raise SystemExit(f"{f} not found: run  modal_jobs/cubicasa_local.py extract  first")
    plans = [p for p in json.loads(gzip.decompress(f.read_bytes())) if "objs" in p and p.get("split", "train") == "train"]
    truth = load_truth()
    print(f"{len(plans)} train plans with annotations")
    for q, classes in QUESTION_CLASSES.items():
        ok = n = 0
        for p in plans:
            gt = truth.get(f"data/cubicasa5k/{p['name']}.png", {}).get(q)
            if gt is None:
                continue
            c = collections.Counter(lab for lab, _ in plan_boxes(p))
            n += 1
            ok += sum(c[k] for k in classes) == gt
        print(f"  label count == Stoa's answer, {q:9}: {ok}/{n} = {ok / max(n, 1):.1%}")


if __name__ == "__main__":
    main()
