"""Build web/benchmark.json from the evaluation runs on disk (shown in the app's Benchmark view).

    uv run python -m stoa_qa.eval --source swiss_dwellings --out eval_out/swiss_full
    uv run python -m stoa_qa.eval --source ifc_models --out eval_out/ifc
    uv run python -m stoa_qa.cubicasa_hybrid
    uv run python -m stoa_qa.report

Only measured numbers go in. Anything estimated is labelled as such.
"""
from __future__ import annotations

import datetime
import json
import re
from pathlib import Path

from .cubicasa_gemini import QUESTIONS, load_truth, split_members
from .cubicasa_hybrid import TILE_QUESTIONS, acc, load_detector, load_sources

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "web" / "benchmark.json"
KEYS = [k for k, _, _ in QUESTIONS]


def _summary(name: str) -> dict | None:
    f = REPO / "eval_out" / name / "summary.json"
    return json.loads(f.read_text()) if f.exists() else None


def _avg_tokens(d: Path) -> tuple[float, float, float, int]:
    tin = tout = cost = n = 0
    for f in d.glob("*.json"):
        u = json.loads(f.read_text()).get("usage") or {}
        if not u:
            continue
        tin += u.get("promptTokenCount", 0)
        tout += u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
        cost += u.get("cost_usd") or 0
        n += 1
    return (tin / n, tout / n, cost / n, n) if n else (0, 0, 0, 0)


def main() -> None:
    groups, notes, kpis = [], [], []
    swiss, ifc = _summary("swiss_full"), _summary("ifc")

    det_rows = []
    if swiss:
        det_rows.append({"label": "Swiss dwellings · CAD (DXF)", "sub": f"{swiss['scored']:,} questions · EN DE FR IT · {swiss['ms_per_question']} ms each · 0 LLM tokens",
                         "pct": 100 * swiss["accuracy"], "kind": ""})
        kpis.append({"value": f"{100 * swiss['accuracy']:.2f}%", "label": f"CAD floor plans · {swiss['scored']:,} questions, 4 languages"})
    if ifc:
        det_rows.append({"label": "IFC models · BIM", "sub": f"{ifc['scored']:,} questions · 9 languages · {ifc['ms_per_question']} ms each · 0 LLM tokens",
                         "pct": 100 * ifc["accuracy"], "kind": ""})
        kpis.append({"value": f"{100 * ifc['accuracy']:.2f}%", "label": f"BIM models · {ifc['scored']:,} questions, 9 languages"})
    if det_rows:
        groups.append({"title": "Computed from geometry — the full question set", "rows": det_rows,
                       "note": "Counts must be exact; areas, lengths and volumes within 2% (Stoa's scoring). No model is involved in these answers."})

    # raster plans, held-out test split only
    truth = load_truth()
    test = sorted(p for p in split_members("test") if p in truth)
    picks_file = Path(__file__).with_name("cubicasa_picks.json")
    picks = json.loads(picks_file.read_text()) if picks_file.exists() else {}
    sources, _, _ = load_sources(picks.get("yolo_run"), picks.get("yolo_thresholds") or {}, picks.get("yolo_iou") or {})
    tiles, _, _ = load_detector(picks.get("tile_run"), TILE_QUESTIONS, picks.get("tile_thresholds") or {}, picks.get("tile_iou") or {})
    if tiles is not None:
        sources["tiles"] = tiles

    def overall(preds: dict) -> tuple[float, int, int]:
        ok = n = 0
        for k in KEYS:
            a, b = acc(preds, truth, test, k)
            ok, n = ok + a, n + b
        return (100 * ok / n if n else 0.0), n, sum(p in preds for p in test)

    raster_rows = []
    hybrid_pct = None
    if picks.get("picks"):
        hybrid = {p: {k: sources[s][p][k] for k, s in picks["picks"].items() if p in sources.get(s, {}) and k in sources[s][p]} for p in test}
        pct, n, _ = overall(hybrid)
        hybrid_pct = pct
        raster_rows.append({"label": "Plumbline hybrid", "sub": f"{n:,} questions · best source per question, chosen on the validation split", "pct": pct, "kind": "ev"})
    g_in, g_out, g_cost, _ = _avg_tokens(REPO / "eval_out/gemini/google__gemma-4-31B-it")
    labels = {"gemma": ("Gemma 4 31B alone", f"{g_in:.0f} + {g_out:.0f} tokens per plan · ${1000 * g_cost / 24:.3f} per 1,000 questions"),
              "network": ("CubiCasa5K network + rules", "pretrained 2019 model · CPU · our post-processing (class-map counts, wet rooms, floor plates)"),
              "gemini": ("Gemini 3.5 Flash alone", ""),
              "yolo": ("Detector alone", "YOLO11s trained on CubiCasa's train-split annotations · one box per counted item")}
    for s in ("yolo", "gemma", "network", "gemini"):
        if s not in sources:
            continue
        pct, n, plans = overall(sources[s])
        if not n:
            continue
        sub = labels[s][1]
        if s == "gemini":
            f_in, f_out, _, _ = _avg_tokens(REPO / "eval_out/gemini/gemini-3.5-flash")
            sub = f"only {plans} plans (free-tier quota ran out) · {f_in:.0f} + {f_out:.0f} tokens per plan"
        raster_rows.append({"label": labels[s][0], "sub": f"{n:,} questions · {sub}", "pct": pct, "kind": "base"})
    if raster_rows:
        groups.append({"title": f"Raster plans — CubiCasa5K held-out test split ({len(test)} plans)", "rows": raster_rows,
                       "note": "Scored only on plans the pretrained network never trained on. When two independent sources agree the answer is right "
                               "far more often, so agreement is shown as the confidence of each answer."})

    # document QA
    log = REPO / "eval_out" / "docqa_opening_design_v2.log"
    if log.exists():
        ok = n = 0
        for a, b in re.findall(r"count questions correct: (\d+)/(\d+)", log.read_text()):
            ok, n = ok + int(a), n + int(b)
        if n:
            groups.append({"title": "Real drawing sets — schedules on PDF sheets (Opening Design)",
                           "rows": [{"label": "Schedule row counts", "sub": f"{n} count questions · Gemma picks the rows, we count them and highlight each one",
                                     "pct": 100 * ok / n, "kind": "ev"}]})

    if swiss and ifc and hybrid_pct is not None:
        n_cubi = sum(1 for line in open(REPO / "junction-dataset/junction-dataset/questions.jsonl", encoding="utf-8") if "/cubicasa5k/" in line)
        est = (swiss["correct"] + ifc["correct"] + hybrid_pct / 100 * n_cubi) / (swiss["scored"] + ifc["scored"] + n_cubi)
        kpis.append({"value": f"{hybrid_pct:.1f}%", "label": "raster plans, held-out test (hybrid)"})
        kpis.append({"value": f"{100 * est:.1f}%", "label": "projected on all 126k scored questions*"})
        notes.append(f"* Projection: measured CAD and BIM results plus the held-out raster rate applied to all {n_cubi:,} raster questions. "
                     "Raster plans are two thirds of the question set, so that is where the next gains are (a detector trained on CubiCasa labels is prepared).")
    notes.append("Answers on CAD and BIM drawings use no LLM at all. Free-form questions use Gemma 4 only to plan a query (about 1,200 tokens, $0.00015); the numbers are still computed from the drawing.")

    OUT.write_text(encoding="utf-8", data=json.dumps({"generated": datetime.date.today().isoformat(), "kpis": kpis, "groups": groups, "notes": notes}, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"kpis": kpis, "groups": [(g["title"], [(r["label"], round(r["pct"], 2)) for r in g["rows"]]) for g in groups]}, indent=1, ensure_ascii=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
