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

from .cubicasa_gemini import QUESTIONS, REPO, load_truth, split_members

KEYS = [k for k, _, _ in QUESTIONS]
YOLO_CLASSES = ["door", "window", "toilet", "sink", "shower", "bathtub"]
YOLO_Q = {"doors": ["door"], "windows": ["window"], "toilets": ["toilet"], "sinks": ["sink"], "baths": ["shower", "bathtub"]}
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 19)]


def load_counts_dir(d: Path) -> dict[str, dict]:
    out = {}
    if d.is_dir():
        for f in d.glob("*.json"):
            r = json.loads(f.read_text())
            if "counts" in r and "path" in r:
                out[r["path"]] = {k: int(v) for k, v in r["counts"].items() if isinstance(v, (int, float))}
    return out


def yolo_counts(boxes: list, thr: dict[str, float]) -> dict[str, int]:
    out = {}
    for q, cls in YOLO_Q.items():
        ids = [YOLO_CLASSES.index(c) for c in cls]
        out[q] = sum(1 for b in boxes if b[0] in ids and b[1] >= thr[q])
    return out


def tune_yolo(val_boxes: dict, truth: dict) -> dict[str, float]:
    thr = {}
    for q in YOLO_Q:
        best = max(THRESHOLDS, key=lambda t: sum(
            yolo_counts(b, {qq: t for qq in YOLO_Q})[q] == truth[p][q]
            for p, b in val_boxes.items() if q in truth.get(p, {})))
        thr[q] = best
    return thr


def acc(preds: dict, truth: dict, plans, key) -> tuple[int, int]:
    ok = n = 0
    for p in plans:
        if key in truth.get(p, {}) and p in preds and key in preds[p]:
            n += 1
            ok += preds[p][key] == truth[p][key]
    return ok, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-run", default="y11s-1280")
    a = ap.parse_args()
    truth = load_truth()
    split = {s: sorted(p for p in split_members(s) if p in truth) for s in ("val", "test")}

    sources: dict[str, dict] = {}
    net = {}
    for f in (REPO / "third_party/cubicasa_preds").glob("*.json"):
        r = json.loads(f.read_text())
        if "counts" in r:
            net[r["path"]] = r["counts"]
    sources["network"] = net
    sources["gemma"] = load_counts_dir(REPO / "eval_out/gemini/google__gemma-4-31B-it")
    sources["gemini"] = load_counts_dir(REPO / "eval_out/gemini/gemini-3.5-flash")

    ydir = REPO / "eval_out/yolo" / a.yolo_run
    if (ydir / "val.json").exists() and (ydir / "test.json").exists():
        vb = {f"data/cubicasa5k/{k}.png": v for k, v in json.loads((ydir / "val.json").read_text()).items()}
        tb = {f"data/cubicasa5k/{k}.png": v for k, v in json.loads((ydir / "test.json").read_text()).items()}
        thr = tune_yolo(vb, truth)
        print("yolo thresholds (tuned on val):", thr)
        sources["yolo"] = {p: yolo_counts(b, thr) for p, b in {**vb, **tb}.items()}

    cov = {s: (sum(p in d for p in split["val"]), sum(p in d for p in split["test"])) for s, d in sources.items()}
    print("coverage (val, test plans):", cov)
    full = [s for s, (v, t) in cov.items() if v >= 0.9 * len(split["val"]) and t >= 0.9 * len(split["test"])]

    print(f"\n{'question':<11}" + "".join(f"{s:>16}" for s in sources) + f"{'HYBRID':>10}  picked (on val)")
    hybrid: dict[str, dict] = collections.defaultdict(dict)
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
        for p in split["test"]:
            if p in sources[best] and k in sources[best][p]:
                hybrid[p][k] = sources[best][p][k]
        ok, n = acc(hybrid, truth, split["test"], k)
        tot_ok, tot_n = tot_ok + ok, tot_n + n
        print(row + f"{ok / max(n, 1):>10.1%}  {best}")
    if tot_n:
        print(f"\nHYBRID on held-out test: {tot_ok}/{tot_n} = {tot_ok / tot_n:.1%}  (each question counted once; EN and FI share answers)")

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
