"""Compare every cubicasa5k source per question and build the hybrid, fairly.

Sources (each optional; missing ones are skipped):
  network  pretrained CubiCasa5K network   third_party/cubicasa_preds/<plan>.json
  gemma    Gemma 4 31B via Featherless     eval_out/gemini/google__gemma-4-31B-it/<plan>.json
  gemini   Gemini 3.5 Flash                eval_out/gemini/gemini-3.5-flash/<plan>.json
  yolo     our detector (Modal)            eval_out/yolo/<run>/{val,test}.json (boxes; thresholds tuned on val)

Selection is done on CubiCasa's VAL split, numbers are reported on the held-out TEST split.
Agreement between independent sources is reported as a calibrated confidence signal.

    uv run python -m stoa_qa.cubicasa_hybrid [--yolo-run y11s-1280]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from . import cubicasa_refine as refine
from .cubicasa_gemini import QUESTIONS, REPO, load_truth, split_members
from .cubicasa_rules import CLASSES as YOLO_CLASSES
from .cubicasa_rules import QUESTION_CLASSES as YOLO_Q
from .cubicasa_rules import select_boxes

KEYS = [k for k, _, _ in QUESTIONS]
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 19)]


def load_counts_dir(d: Path) -> dict[str, dict]:
    out = {}
    if d.is_dir():
        for f in d.glob("*.json"):
            r = json.loads(f.read_text())
            if "counts" in r and "path" in r:
                out[r["path"]] = {k: int(v) for k, v in r["counts"].items() if isinstance(v, (int, float))}
    return out


IOUS = [0.3, 0.5, 0.7, 1.0]  # cross-class duplicate suppression; 1.0 = off


TILE_QUESTIONS = ("doors", "windows")  # the tiled detector (modal_jobs/cubicasa_tiles.py) only has these classes


def yolo_counts(boxes: list, thr: dict[str, float], iou: dict[str, float] | None = None, questions=None) -> dict[str, int]:
    return {q: len(select_boxes(boxes, q, thr[q], (iou or {}).get(q, 1.0))) for q in (questions or YOLO_Q)}


def tune_yolo(val_boxes: dict, truth: dict, questions=None) -> tuple[dict[str, float], dict[str, float]]:
    """Per question: the score threshold and duplicate-overlap cut that are exact most often on VAL."""
    thr, iou = {}, {}
    for q in (questions or YOLO_Q):
        plans = [(b, truth[p][q]) for p, b in val_boxes.items() if q in truth.get(p, {})]
        # ties go to the simpler setting: no suppression, then the threshold closest to 0.5
        best = max(((t, u) for t in THRESHOLDS for u in IOUS),
                   key=lambda tu: (sum(len(select_boxes(b, q, tu[0], tu[1])) == g for b, g in plans), tu[1], -abs(tu[0] - 0.5)))
        thr[q], iou[q] = best
    return thr, iou


def acc(preds: dict, truth: dict, plans, key) -> tuple[int, int]:
    ok = n = 0
    for p in plans:
        if key in truth.get(p, {}) and p in preds and key in preds[p]:
            n += 1
            ok += preds[p][key] == truth[p][key]
    return ok, n


def _boxes(run: str, split: str) -> dict:
    f = REPO / "eval_out/yolo" / run / f"{split}.json"
    return {f"data/cubicasa5k/{k}.png": v for k, v in json.loads(f.read_text()).items()} if f.exists() else {}


def load_detector(run: str | None, questions=None, thr: dict | None = None, iou: dict | None = None, truth: dict | None = None):
    """(per-plan counts, thresholds, overlap cuts) for one detector run; tuned on VAL when `thr` is not given."""
    vb, tb = (_boxes(run, "val"), _boxes(run, "test")) if run else ({}, {})
    if not vb or not tb:
        return None, {}, {}
    if not thr:
        thr, iou = tune_yolo(vb, truth, questions)
    return {p: yolo_counts(b, thr, iou, questions) for p, b in {**vb, **tb}.items()}, thr, iou or {}


def load_sources(yolo_run: str | None, thr: dict | None = None, iou: dict | None = None, truth: dict | None = None):
    """Every source's per-plan counts, as the app computes them. Returns (sources, yolo thresholds, yolo overlap cuts).

    Without `thr`, the detector's thresholds are tuned here on the VAL split (needs `truth`)."""
    sources: dict[str, dict] = {}
    seg = {}
    seg_file = REPO / "eval_out/seg_components.jsonl"
    if seg_file.exists():
        for line in seg_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                seg[rec["path"]] = rec
    net = {}
    for f in (REPO / "third_party/cubicasa_preds").glob("*.json"):
        r = json.loads(f.read_text())
        if "counts" in r:  # the counts the app serves: network + refine rules (+ class-map components when cached)
            net[r["path"]] = refine.apply(r, seg.get(r["path"]))
    sources["network"] = net
    sources["gemma"] = load_counts_dir(REPO / "eval_out/gemini/google__gemma-4-31B-it")
    sources["gemini"] = load_counts_dir(REPO / "eval_out/gemini/gemini-3.5-flash")

    counts, thr, iou = load_detector(yolo_run, None, thr, iou, truth)
    if counts is not None:
        sources["yolo"] = counts
    return sources, thr, iou


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-run", default="local-y11s-13c")
    ap.add_argument("--tile-run", default=None, help="tiled door/window detector run (modal_jobs/cubicasa_tiles.py)")
    a = ap.parse_args()
    truth = load_truth()
    split = {s: sorted(p for p in split_members(s) if p in truth) for s in ("val", "test")}

    sources, thr, iou = load_sources(a.yolo_run, truth=truth)
    if "yolo" in sources:
        print("yolo thresholds (tuned on val):", thr)
        print("yolo duplicate-overlap cut (tuned on val):", iou)
    tcounts, tthr, tiou = load_detector(a.tile_run, TILE_QUESTIONS, truth=truth)
    if tcounts is not None:
        sources["tiles"] = tcounts
        print("tile thresholds / overlap cuts (tuned on val):", tthr, tiou)

    cov = {s: (sum(p in d for p in split["val"]), sum(p in d for p in split["test"])) for s, d in sources.items()}
    print("coverage (val, test plans):", cov)
    full = [s for s, (v, t) in cov.items() if v >= 0.9 * len(split["val"]) and t >= 0.9 * len(split["test"])]

    print(f"\n{'question':<11}" + "".join(f"{s:>16}" for s in sources) + f"{'HYBRID':>10}  picked (on val)")
    hybrid: dict[str, dict] = collections.defaultdict(dict)
    picked: dict[str, str] = {}
    tot_ok = tot_n = 0
    for k in KEYS:
        row = f"{k:<11}"
        for s, d in sources.items():
            ok, n = acc(d, truth, split["test"], k)
            row += f"{(f'{ok / n:.0%} ({n})' if n else '-'):>16}"
        cands = [s for s in full if acc(sources[s], truth, split["val"], k)[1]]
        if not cands:
            print(row + f"{'-':>10}")
            continue
        best = max(cands, key=lambda s: acc(sources[s], truth, split["val"], k)[0] / max(1, acc(sources[s], truth, split["val"], k)[1]))
        picked[k] = best
        for p in split["test"]:
            if p in sources[best] and k in sources[best][p]:
                hybrid[p][k] = sources[best][p][k]
        ok, n = acc(hybrid, truth, split["test"], k)
        tot_ok, tot_n = tot_ok + ok, tot_n + n
        print(row + f"{ok / max(n, 1):>10.1%}  {best}")
    if tot_n:
        print(f"\nHYBRID on held-out test: {tot_ok}/{tot_n} = {tot_ok / tot_n:.1%}  (each question counted once; EN and FI share answers)")
    if picked:
        out = Path(__file__).with_name("cubicasa_picks.json")
        out.write_text(json.dumps({"picks": picked, "selected_on": "CubiCasa5K val split", "yolo_run": a.yolo_run if "yolo" in sources else None,
                                   "yolo_thresholds": thr, "yolo_iou": iou,
                                   "tile_run": a.tile_run if "tiles" in sources else None,
                                   "tile_thresholds": tthr, "tile_iou": tiou,
                                   "test_accuracy": round(tot_ok / max(tot_n, 1), 4), "test_questions": tot_n}, indent=1) + "\n")
        print(f"picks -> {out}")

    # agreement as confidence: hybrid answer confirmed by at least one other full-coverage source
    agree = collections.Counter()
    for p in split["test"]:
        for k, v in hybrid.get(p, {}).items():
            if k not in truth.get(p, {}):
                continue
            others = [sources[s][p][k] for s in full if p in sources[s] and k in sources[s][p]]
            confirmed = sum(o == v for o in others) >= 2  # the picked source itself + one independent source
            agree[(confirmed, v == truth[p][k])] += 1
    for c in (True, False):
        n = agree[(c, True)] + agree[(c, False)]
        if n:
            print(f"  {'sources agree   ' if c else 'sources disagree'}: {n / max(1, tot_n):5.1%} of answers, accuracy {agree[(c, True)] / n:.1%}")


if __name__ == "__main__":
    main()
