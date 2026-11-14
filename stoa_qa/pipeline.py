"""Answer pipeline: a crew of specialists, cheapest trustworthy one first, and an auditor on every answer.

    0. navigator      room graph: routes, neighbours, openings     0 LLM tokens (stoa_qa.graph)
    0b. MEP engineer  pipes, ducts, terminals, systems, ports      0 LLM tokens (stoa_qa.mep)
    1. surveyor       geometry / BIM / precomputed detections      0 LLM tokens, exact
    2. planner        Gemma plans a query, we execute it on facts  ~1k tokens, numbers still computed
    3. clerk          rows of a PDF sheet's own text               Gemma picks rows, we count
    4. vision         Gemma reads the sheet image                  estimate, flagged as such
    +  auditor        independent checks on the answer             0 LLM tokens (stoa_qa.audit)

Every result is turned into an overlay in the coordinate space of the sheet the user is looking at.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import audit, catalog, cubicasa, docqa, explain, graph, legend, llm, mep, planner, router, sheets, vision, xref
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

# Free-form questions only go to the template engines when they have the templates' shape (a count or an area) and
# carry no qualifier those engines would silently ignore ("windows facing south" is not "windows").
COUNT_Q = re.compile(r"^\s*(?:how many|wie viele|wieviele|combien|quant[ie]|quanti|hoeveel|hvor mange|ile|câte|kui mitu|cuánt[oa]s|"
                     r"number of|count|total number)\b", re.I)
AREA_Q = re.compile(r"\b(?:area|fläche|surface|superficie|oppervlakte|areal|pindala|powierzchni|suprafa|volume|volumen|length|länge|"
                    r"m²|m2|m³|m3)\b", re.I)
IGNORED_BY_TEMPLATES = re.compile(
    r"\b(?:fac(?:e|es|ing)|orient\w*|north|south|east|west|material\w*|made of|built (?:of|from)|construct\w*|outside|exterior|external|"
    r"type|types|kind|colou?r|scale|height|tall|thick\w*|wide|width|cost|price|fire|rating|insulat\w*|year|date|architect|author|who|"
    r"open(?:s)? (?:on)?to|connect\w*|adjacent|next to)", re.I)
# questions about what is printed on the sheet (title block, north arrow, notes): the sheet readers get a turn on every format
SHEET_INFO = re.compile(
    r"\b(?:scale|maßstab|massstab|échelle|north|nord|norden|orientation|architect\w*|author|drawn by|designed by|designer|"
    r"who (?:drew|made|designed|created|prepared)|date[ds]?|when was|revision|project (?:name|number|title)|client|"
    r"title block|sheet (?:title|number|name)|address|legend|notes?|stamp)\b", re.I)
YES_NO = re.compile(r"^\s*(?:is|are|does|do|has|have|can|was|were|ist|sind|gibt es|hat|y a-t-il|est-ce|c'è|ci sono)\b", re.I)


def template_shaped(question: str) -> bool:
    return bool((COUNT_Q.search(question) or AREA_Q.search(question)) and not IGNORED_BY_TEMPLATES.search(question))


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
    trace: list[dict] = field(default_factory=list)   # the crew: who was asked, in order, and what came of it


# The crew. Each member is one engine below; the general contractor (solve) asks the cheapest one that can
# answer, moves on when it declines, and the auditor checks whatever answer comes back (stoa_qa.audit).
CREW = {
    "graph": ("Navigator", "room graph: routes, neighbours, doors between rooms"),
    "geometry": ("Quantity surveyor", "counts and areas from CAD geometry"),
    "bim": ("Quantity surveyor", "counts and quantities from the BIM model"),
    "detection": ("Quantity surveyor", "counts from detectors on the scanned plan"),
    "mep": ("MEP engineer", "pipes, ducts, terminals, systems and their connections"),
    "xref": ("Reference clerk", "follows callouts like 6A/A-401 to the detail they point at"),
    "planner": ("Planner", "an LLM writes one query; the numbers are computed"),
    "document": ("Document clerk", "rows of the sheet's own text"),
    "vision": ("Vision reader", "an LLM reads the image (estimate)"),
}


def _usage(u) -> list[dict]:
    return [u.to_dict()] if u is not None else []


def primary_raster(d: Drawing) -> tuple[str, str, int] | None:
    for s in d.sheets:
        if s.kind in ("image", "pdf") and not s.path.lower().endswith(".svg"):
            return s.kind, s.path, s.page
    return None


def solve(d: Drawing, question: str, sheet: tuple[str, int] | None = None) -> Result:
    t0 = time.perf_counter()
    question = question.strip()
    exact = d.source in TEMPLATED and any(q["question"].strip() == question for q in catalog.questions_for(d))
    structured = d.source in router.SOLVERS
    topology = not exact and graph.intent_of(question) is not None
    services = not exact and d.source in ("ifc_models", "ifc_bench") and mep.intent_of(question) is not None
    simple = (d.source in TEMPLATED and not COMPLEX.search(question) and len(question.split()) <= 22 and not topology and not services
              and (d.source == "cubicasa5k" or template_shaped(question)))
    sheet_info = not exact and SHEET_INFO.search(question) is not None
    usage: list[dict] = []
    trace: list[dict] = []
    fallback: Answer | None = None

    def asked(engine: str, a: Answer | None, t: float, tokens: int = 0) -> None:
        who, role = CREW[engine]
        outcome = "answered" if a is not None and a.value is not None else "declined"
        why = "" if a is None else (a.method if outcome == "declined" else "")
        trace.append({"agent": who, "engine": engine, "role": role, "outcome": outcome, "note": why[:160],
                      "ms": round((time.perf_counter() - t) * 1000, 1), "tokens": tokens})

    def done(a: Answer, engine: str, note: str) -> Result:
        return Result(a, engine, note, usage, time.perf_counter() - t0, trace)

    def tokens(u) -> int:
        return (u.tokens_in + u.tokens_out) if u is not None else 0

    # 0. services: pipes, ducts, terminals and systems in IFC models (no LLM); MEP wording goes here first
    if services:
        t = time.perf_counter()
        a = mep.answer(d.ask_path, question)
        asked("mep", a, t)
        if a is not None and a.value is not None:
            return done(a, "mep", "Read from the model's distribution elements, systems and ports")

    # 0b. topology: how rooms connect, traversed on the room graph (no LLM)
    if topology:
        t = time.perf_counter()
        a = graph.answer(d.ask_path, question)
        asked("graph", a, t)
        if a is not None and a.value is not None:
            return done(a, "graph", "Traversed the room graph built from the drawing's geometry")
        fallback = a

    # 1. deterministic
    engine = ENGINE_LABEL.get(d.source, ("geometry", ""))[0]
    if structured and (exact or simple):
        t = time.perf_counter()
        a = router.answer(d.ask_path, question)
        if a.value is None and d.source == "cubicasa5k" and cubicasa.question_key(question) and llm.available():
            usage += _live_cubicasa_counts(d)
            a = router.answer(d.ask_path, question)
        asked(engine, a, t)
        if a.value is not None:
            return done(a, *ENGINE_LABEL[d.source])
        fallback = a

    # 2. planner over fact tables
    if planner.tables_for(d.ask_path) is not None and llm.available():
        t = time.perf_counter()
        a, u = planner.answer(d.ask_path, question)
        usage += _usage(u)
        asked("planner", a, t, tokens(u))
        scoped = fallback is not None and fallback.intent == "scoped"  # the template engine asked "which room?": keep that question
        if a.value is not None:
            return done(a, "planner", f"Query planned by {llm.MODEL.split('/')[-1]}, executed on exact data")
        if a.intent == "unanswerable" and not sheet_info:
            if scoped:
                return done(fallback, "none", "")
            return done(a, "planner", f"Query planned by {llm.MODEL.split('/')[-1]}, executed on exact data")
        if not scoped:
            fallback = a if a.intent == "unanswerable" else (fallback or a)  # the sheet may still show it (title block, north arrow)

    # 1b. template engine as a last structured resort (question had 'complex' wording but nothing else answered)
    if structured and fallback is None and (d.source == "cubicasa5k" or template_shaped(question)):
        t = time.perf_counter()
        a = router.answer(d.ask_path, question)
        asked(engine, a, t)
        if a.value is not None:
            return done(a, *ENGINE_LABEL[d.source])
        fallback = a

    # 2b. cross-sheet references on a PDF sheet: callouts followed through the project's sheet index (no LLM)
    pdf = next((s for s in d.sheets if s.kind == "pdf"), None)
    here = sheet if sheet and sheet[0].lower().endswith(".pdf") else ((pdf.path, pdf.page) if pdf is not None else None)
    if here is not None and xref.intent_of(question):
        t = time.perf_counter()
        a = xref.answer(here[0], question, here[1])
        asked("xref", a, t)
        if a is not None and a.value is not None:
            return done(a, "xref", "Callouts read from the sheet's text, resolved through the project's sheet index")
        fallback = fallback or a

    # 3. the sheet's own text (schedules, title blocks) when it is a real PDF; on CAD/BIM drawings only for sheet-info questions
    if pdf is not None and (not structured or sheet_info) and llm.available() and docqa.has_text(pdf.path, pdf.page):
        t = time.perf_counter()
        a, u = docqa.ask(pdf.path, question, pdf.page)
        usage += _usage(u)
        asked("document", a, t, tokens(u))
        if a.value is not None:
            return done(a, "document", f"Rows of the sheet's text selected by {llm.MODEL.split('/')[-1]}, counted by us")
        fallback = fallback or a

    # 3b. an uploaded raster sheet: the same document clerk over its text, read at full resolution (stoa_qa.sheet_ocr)
    rows = _upload_rows(d) if pdf is None and llm.available() else None
    if rows is not None and len(rows) >= 8:
        t = time.perf_counter()
        a, u = docqa.ask(d.ask_path, question, rows=rows)
        usage += _usage(u)
        asked("document", a, t, tokens(u))
        if a.value is not None:
            return done(a, "document", f"Rows of the sheet's text (read by OCR at full resolution) selected by "
                                       f"{llm.MODEL.split('/')[-1]}, counted by us")
        fallback = fallback or a

    # 4. vision on the sheet
    raster = primary_raster(d)
    if raster is not None and llm.available():
        kind, path, page = raster
        t = time.perf_counter()
        img = sheets.raster_path(kind, path, page)
        a, u = vision.ask(img, question, sheets.image_size(kind, path, page), cache_key=f"{path}|{page}")
        usage += _usage(u)
        asked("vision", a, t, tokens(u))
        if a.value is not None:
            return done(a, "vision", f"Read from the sheet by {llm.MODEL.split('/')[-1]}")
        fallback = fallback or a

    a = fallback or Answer(None, None, "unknown", 0.0, "No engine can answer this for this drawing yet.")
    return done(a, "none", "")


def _upload_rows(d: Drawing) -> list[dict] | None:
    """Printed rows of an uploaded raster sheet (OCR, cached per image), None for anything else or without OCR."""
    if not d.ask_path.startswith("uploads/"):
        return None
    from . import cubicasa_live

    image = sheets.safe_path(d.ask_path)
    try:
        return docqa.raster_layout(image, cubicasa_live.image_hash(image))
    except Exception:  # OCR not installed or unreadable image: the vision reader still runs
        return None


def presented(question: str, a: Answer) -> str:
    """The answer as a sentence: yes/no questions answered with a count read as yes/no."""
    if a.value is None:
        return answer_text(a)
    if YES_NO.match(question) and a.unit == "count" and isinstance(a.value, (int, float)) and not isinstance(a.value, bool):
        n = int(a.value)
        return f"Yes — {n} found" if n > 0 else "No — none found"
    return answer_text(a)


def _live_cubicasa_counts(d: Drawing) -> list[dict]:
    """A plan without precomputed predictions and no detector to run here: one Gemma call counts all 12 items, cached.

    cubicasa.answer already tries the YOLO detectors live (stoa_qa.cubicasa_live) before this is reached."""
    from . import cubicasa_gemini as cg

    cache = REPO / "eval_out/gemini" / llm.MODEL.replace("/", "__")
    cache.mkdir(parents=True, exist_ok=True)
    path = str(sheets.safe_path(d.ask_path)) if d.ask_path.startswith("uploads/") else d.ask_path  # DATASET / absolute = absolute
    r = cg.ask_featherless(llm.MODEL, llm.api_key(), path, 1536, cache)
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


# How each marked item was established, for the confidence heatmap:
#   direct    read from CAD geometry, the BIM model or the sheet's own text
#   inferred  counted by a detector or the CubiCasa network from the pixels
#   review    only a vision model saw it: a person signs it off (the web app also marks everything red on an auditor fail)
BASIS_NOTE = {"direct": "read from geometry, the model or the sheet text", "inferred": "detected from the pixels",
              "review": "vision model only: needs a person's sign-off"}


def _raster_basis(ref: str) -> str:
    if ref.startswith("vision"):
        return "review"
    if ref.startswith("L") and ref[1:].isdigit():  # a printed row of the PDF's text (stoa_qa.docqa)
        return "direct"
    return "inferred"  # yolo / tiles / seg / wet / floor / net: detector and network marks


def overlay(d: Drawing, a: Answer) -> dict:
    """Evidence in the coordinate space of the sheet: image pixels, or IFC GlobalIds for SVG storey plans.

    Every item carries `basis` (direct | inferred | review). PDF sheets with a printed legend also get `legends`,
    and an item whose type mark appears in one gets `legend` = [legend index, row index]."""
    out = _overlay(d, a)
    if d.source == "opening_design" and out["space"] == "px":
        try:
            lgs = legend.for_drawing(d)
        except Exception:  # a broken PDF must not break the answer
            lgs = []
        if lgs:
            out["legends"] = lgs
            top = legend.match(lgs, str(a.value or ""))  # the mark the answer names (e.g. door type B3) wins
            if top:
                out["legend"] = list(top)
            top_marks = set(legend.MARK.findall(str(a.value or "").upper()))
            for it in out["items"]:
                label = str(it.get("label") or "").upper()
                m = top if top and top_marks & set(legend.MARK.findall(label)) else legend.match(lgs, label)
                if m:
                    it["legend"] = list(m)
    return out


def _overlay(d: Drawing, a: Answer) -> dict:
    items: list[dict] = []
    if d.source == "swiss_dwellings":
        stem = Path(d.ask_path).stem
        path_px = None
        for e in a.evidence:
            if not e.coords:
                continue
            if e.kind == "route":  # the walked line of a graph route: drawn under the numbered items
                path_px = sheets.swiss_to_px(stem, e.coords)
                continue
            px = sheets.swiss_to_px(stem, e.coords)
            if px is None:
                continue
            kind = "point" if len(px) == 1 else "poly"
            box = _bbox(px) if kind == "poly" else [px[0][0] - 40, px[0][1] - 40, px[0][0] + 40, px[0][1] + 40]
            items.append({"kind": kind, "pts": px, "bbox": box, "label": e.label, "value": e.value, "id": e.viewer_id,
                          "basis": "direct"})
        out = {"space": "px", "items": _number(items)}
        if path_px:
            out["path"] = path_px
        return out
    if any(e.kind == "ifc" for e in a.evidence):
        guids = [e.ref for e in a.evidence if e.kind == "ifc"]
        for e in a.evidence:
            if e.kind == "ifc" and e.ref:
                items.append({"kind": "guid", "guid": e.ref, "label": e.label, "value": e.value, "id": e.viewer_id,
                              "basis": "direct"})
        hits = []
        for s in d.sheets:
            if s.kind == "svg":
                n = len(sheets.svg_guids(s.path).intersection(guids))
                if n:
                    hits.append({"path": s.path, "hits": n})
        hits.sort(key=lambda h: -h["hits"])
        for it in items:  # the storey plan each element is drawn on, so a click can open exactly that sheet
            it["sheet"] = next((h["path"] for h in hits if it["guid"] in sheets.svg_guids(h["path"])), None)
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
        items.append({"kind": "poly", "pts": pts, "bbox": _bbox(pts), "label": e.label, "value": e.value, "id": None,
                      "basis": _raster_basis(str(e.ref or ""))})
    return {"space": "px", "items": _number(items)}


def _number(items: list[dict]) -> list[dict]:
    for i, it in enumerate(items):
        it["n"] = i + 1
    return items


def respond(d: Drawing, question: str, sheet: tuple[str, int] | None = None) -> dict:
    r = solve(d, question, sheet)
    a = r.answer
    t = time.perf_counter()
    checks, a.confidence = audit.run(d, question, a, r.engine)
    if a.value is not None:
        r.trace.append({"agent": "Auditor", "engine": "audit", "role": "independent checks on the answer",
                        "outcome": "checked", "note": f"{len(checks)} check(s)", "ms": round((time.perf_counter() - t) * 1000, 1), "tokens": 0})
    gold = next((q for q in catalog.questions_for(d) if q["question"].strip() == question.strip()), None)
    ov = overlay(d, a)
    tokens = sum(u["tokens_in"] + u["tokens_out"] for u in r.usage)
    cost = sum(u["cost_usd"] for u in r.usage)
    out = {
        "question": question,
        "text": presented(question, a),
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
        "crew": r.trace,
        "audit": [c.to_dict() for c in checks],
        "references": getattr(a, "references", None),  # callout -> target detail pairs (stoa_qa.xref)
        "unanswered": (why := explain.explain(d, question, a, r.trace, llm.available())),  # why not, what was checked, what to ask instead
    }
    if why:
        out["text"] = why["headline"]
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as fp:
            fp.write(json.dumps({"t": time.time(), "id": d.id, "q": question, "text": out["text"], "engine": r.engine,
                                 "ms": out["latency_ms"], "tokens": tokens, "cost": cost, "correct": out["correct"]},
                                ensure_ascii=False) + "\n")
    except OSError:
        pass
    return out
