"""CubiCasa5K annotation classes -> detector classes, and a check of those rules against Stoa's answers.

Stdlib only: this module is also imported inside the Modal job that writes the YOLO labels.

    modal run modal_jobs/cubicasa_detector.py::extract   # writes eval_out/cubicasa_objects.json.gz
    uv run python -m stoa_qa.cubicasa_rules              # how often do label counts equal Stoa's answers?
"""
from __future__ import annotations

CLASSES = ["door", "window", "toilet", "sink", "shower", "bathtub", "floor"]
QUESTION_CLASSES = {"doors": ["door"], "windows": ["window"], "toilets": ["toilet"], "sinks": ["sink"],
                    "baths": ["shower", "bathtub"], "floors": ["floor"]}


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
        if k in ("Shower", "ShowerCab"):
            return "shower"
        if k.startswith("Bathtub"):
            return "bathtub"
    return None


def plan_boxes(plan: dict) -> list[tuple[str, list[float]]]:
    """(class, [x0, y0, x1, y1]) for one plan record from cubicasa_detector.extract (original image pixels)."""
    out = [(lab, o["b"]) for o in plan["objs"] if (lab := label_of(o["c"])) and o.get("b")]
    out += [("floor", f["b"]) for f in plan.get("floors", []) if f.get("b")]
    return out


def main() -> None:
    import collections
    import gzip
    import json
    from pathlib import Path

    from .cubicasa_gemini import load_truth

    f = Path(__file__).resolve().parent.parent / "eval_out" / "cubicasa_objects.json.gz"
    if not f.exists():
        raise SystemExit(f"{f} not found: run  modal run modal_jobs/cubicasa_detector.py::extract  first")
    plans = [p for p in json.loads(gzip.decompress(f.read_bytes())) if "objs" in p]
    truth = load_truth()
    print(f"{len(plans)} plans with annotations")
    for q, classes in QUESTION_CLASSES.items():
        ok = n = 0
        for p in plans:
            gt = truth.get(f"data/cubicasa5k/{p['name']}.png", {}).get(q)
            if gt is None:
                continue
            c = collections.Counter(lab for lab, _ in plan_boxes(p))
            n += 1
            ok += sum(c[k] for k in classes) == gt
        print(f"  label count == Stoa's answer, {q:8}: {ok}/{n} = {ok / max(n, 1):.1%}")


if __name__ == "__main__":
    main()
