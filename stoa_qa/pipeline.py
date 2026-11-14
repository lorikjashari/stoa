"""Answer pipeline: cheapest trustworthy engine first.

    1. deterministic  geometry / BIM / precomputed detections      0 LLM tokens, exact
    2. planner        Gemma plans a query, we execute it on facts  ~1k tokens, numbers still computed
    3. vision         Gemma reads the sheet image                  estimate, flagged as such

Every result is turned into an overlay in the coordinate space of the sheet the user is looking at.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import catalog, cubicasa, docqa, llm, planner, router, sheets, vision
from .catalog import Drawing
from .core import Answer, fmt_value

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "eval_out" / "ask_log.jsonl"
TEMPLATED = {"swiss_dwellings", "ifc_models", "cubicasa5k"}  # sources whose dataset questions follow fixed templates

# wording that needs filtering / ranking / grouping: beyond the fixed question templates
COMPLEX = re.compile(
    r"\b(which|most|least|largest|biggest|smallest|average|mean|each|per|more than|less than|greater|bigger|smaller|"
    r"wider|widest|narrowest|tallest|highest|lowest|list|compare|percentage|ratio|between|at least|at most|"
    r"how big|how large|how wide|software|who|when|what kind|what type)\b", re.I)

ENGINE_LABEL = {
    "swiss_dwellings": ("geometry", "Computed from CAD geometry (DXF)"),
    "ifc_models": ("bim", "Computed from the BIM model (IFC)"),
    "ifc_bench": ("bim", "Computed from the BIM model (IFC)"),
    "cubicasa5k": ("detection", "Counted by detectors and vision models, cross-checked"),
}


@dataclass
class Result:
    answer: Answer
    engine: str = "none"
    engine_note: str = ""
    usage: list[dict] = field(default_factory=list)
    seconds: float = 0.0


def _usage(u) -> list[dict]:
    return [u.to_dict()] if u is not None else []


def primary_raster(d: Drawing) -> tuple[str, str, int] | None:
    for s in d.sheets:
        if s.kind in ("image", "pdf") and not s.path.lower().endswith(".svg"):
            return s.kind, s.path, s.page
    return None


def solve(d: Drawing, question: str) -> Result:
    t0 = time.perf_counter()
    question = question.strip()
    exact = d.source in TEMPLATED and any(q["question"].strip() == question for q in catalog.questions_for(d))
    structured = d.source in router.SOLVERS
    simple = d.source in TEMPLATED and not COMPLEX.search(question) and len(question.split()) <= 22
    usage: list[dict] = []
    fallback: Answer | None = None

    def done(a: Answer, engine: str, note: str) -> Result:
        return Result(a, engine, note, usage, time.perf_counter() - t0)

    # 1. deterministic
    if structured and (exact or simple):
        a = router.answer(d.ask_path, question)
        if a.value is None and d.source == "cubicasa5k" and cubicasa.question_key(question) and llm.available():
            usage += _live_cubicasa_counts(d)
            a = router.answer(d.ask_path, question)
        if a.value is not None:
            return done(a, *ENGINE_LABEL[d.source])
        fallback = a

    # 2. planner over fact tables
    if planner.tables_for(d.ask_path) is not None and llm.available():
        a, u = planner.answer(d.ask_path, question)
        usage += _usage(u)
        if a.value is not None or a.intent == "unanswerable":
            return done(a, "planner", f"Query planned by {llm.MODEL.split('/')[-1]}, executed on exact data")
        fallback = fallback or a

    # 1b. template engine as a last structured resort (question had 'complex' wording but nothing else answered)
    if structured and fallback is None:
        a = router.answer(d.ask_path, question)
        if a.value is not None:
            return done(a, *ENGINE_LABEL[d.source])
        fallback = a

    # 3. the sheet's own text (schedules, title blocks) when it is a real PDF
    pdf = next((s for s in d.sheets if s.kind == "pdf"), None)
    if pdf is not None and not structured and llm.available() and docqa.has_text(pdf.path, pdf.page):
        a, u = docqa.ask(pdf.path, question, pdf.page)
        usage += _usage(u)
        if a.value is not None:
            return done(a, "document", f"Rows of the sheet's text selected by {llm.MODEL.split('/')[-1]}, counted by us")
        fallback = fallback or a

    # 4. vision on the sheet
    raster = primary_raster(d)
    if raster is not None and llm.available():
        kind, path, page = raster
        img = sheets.raster_path(kind, path, page)
        a, u = vision.ask(img, question, sheets.image_size(kind, path, page), cache_key=f"{path}|{page}")
        usage += _usage(u)
        if a.value is not None:
            return done(a, "vision", f"Read from the sheet by {llm.MODEL.split('/')[-1]}")
        fallback = fallback or a

    a = fallback or Answer(None, None, "unknown", 0.0, "No engine can answer this for this drawing yet.")
    return done(a, "none", "")


def _live_cubicasa_counts(d: Drawing) -> list[dict]:
    """A plan without precomputed predictions: one Gemma call counts all 12 items, cached for later questions."""
    from . import cubicasa_gemini as cg

    cache = REPO / "eval_out/gemini" / llm.MODEL.replace("/", "__")
    cache.mkdir(parents=True, exist_ok=True)
    r = cg.ask_featherless(llm.MODEL, llm.api_key(), d.ask_path, 1536, cache)
    u = r.get("usage") or {}
    if "error" in r or not u:
        return []
    return [{"model": llm.MODEL, "tokens_in": u.get("promptTokenCount", 0), "tokens_out": u.get("candidatesTokenCount", 0),
             "cost_usd": round(u.get("cost_usd") or 0.0, 6), "seconds": r.get("seconds", 0)}]


# ------------------------------------------------------------------ presentation

def answer_text(a: Answer) -> str:
    return "I can't answer that from this drawing." if a.value is None else fmt_value(a.value, a.unit)


def is_correct(expected, value, qtype: str | None) -> bool | None:
    if expected is None or value is None or qtype in ("text", "other"):
        return None
    try:
        e, v = float(expected), float(value)
    except (TypeError, ValueError):
        return None
    if qtype == "count":
        return round(v) == e
    return e != 0 and abs(v - e) / abs(e) <= 0.02


def _bbox(pts: list[list[float]]) -> list[float]:
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def overlay(d: Drawing, a: Answer) -> dict:
    """Evidence in the coordinate space of the sheet: image pixels, or IFC GlobalIds for SVG storey plans."""
    items: list[dict] = []
    if d.source == "swiss_dwellings":
        stem = Path(d.ask_path).stem
        for e in a.evidence:
            if not e.coords:
                continue
            px = sheets.swiss_to_px(stem, e.coords)
            if px is None:
                continue
            kind = "point" if len(px) == 1 else "poly"
            box = _bbox(px) if kind == "poly" else [px[0][0] - 40, px[0][1] - 40, px[0][0] + 40, px[0][1] + 40]
            items.append({"kind": kind, "pts": px, "bbox": box, "label": e.label, "value": e.value, "id": e.viewer_id})
        return {"space": "px", "items": _number(items)}
    if any(e.kind == "ifc" for e in a.evidence):
        guids = [e.ref for e in a.evidence if e.kind == "ifc"]
        for e in a.evidence:
            if e.kind == "ifc":
                items.append({"kind": "guid", "guid": e.ref, "label": e.label, "value": e.value, "id": e.viewer_id})
        hits = []
        for s in d.sheets:
            if s.kind == "svg":
                n = len(sheets.svg_guids(s.path).intersection(guids))
                if n:
                    hits.append({"path": s.path, "hits": n})
        hits.sort(key=lambda h: -h["hits"])
        return {"space": "guid", "items": _number(items), "sheets": hits}
    for e in a.evidence:  # raster: detector boxes, network polygons, vision boxes
        if not e.coords:
            continue
        if e.kind == "box" and len(e.coords) == 2:
            (x0, y0), (x1, y1) = e.coords
            pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        else:
            pts = e.coords
        if len(pts) < 3:
            continue
        items.append({"kind": "poly", "pts": pts, "bbox": _bbox(pts), "label": e.label, "value": e.value, "id": None})
    return {"space": "px", "items": _number(items)}


def _number(items: list[dict]) -> list[dict]:
    for i, it in enumerate(items):
        it["n"] = i + 1
    return items


def respond(d: Drawing, question: str) -> dict:
    r = solve(d, question)
    a = r.answer
    gold = next((q for q in catalog.questions_for(d) if q["question"].strip() == question.strip()), None)
    ov = overlay(d, a)
    tokens = sum(u["tokens_in"] + u["tokens_out"] for u in r.usage)
    cost = sum(u["cost_usd"] for u in r.usage)
    out = {
        "question": question,
        "text": answer_text(a),
        "value": a.value,
        "unit": a.unit,
        "intent": a.intent,
        "method": a.method,
        "confidence": a.confidence,
        "engine": {"kind": r.engine, "note": r.engine_note, "llm_tokens": tokens, "cost_usd": round(cost, 6),
                   "models": sorted({u["model"].split("/")[-1] for u in r.usage})},
        "latency_ms": round(r.seconds * 1000, 1),
        "expected": gold["answer"] if gold else None,
        "correct": is_correct(gold["answer"], a.value, gold["type"]) if gold else None,
        "overlay": ov,
        "ids": [it["id"] for it in ov["items"] if it.get("id") is not None],
    }
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as fp:
            fp.write(json.dumps({"t": time.time(), "id": d.id, "q": question, "text": out["text"], "engine": r.engine,
                                 "ms": out["latency_ms"], "tokens": tokens, "cost": cost, "correct": out["correct"]},
                                ensure_ascii=False) + "\n")
    except OSError:
        pass
    return out
