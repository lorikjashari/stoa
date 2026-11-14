"""When no engine can answer: say why in plain words, say what was checked, and offer questions that will work.

    category      not_in_data   the drawing / model does not carry this information
                  not_understood the question did not map onto anything the engines read
                  needs_llm     free-form wording needs the language model, which is not configured
                  failed        a reader broke (timeout, malformed reply)
    detail        what this kind of drawing does contain, plus the most specific note an engine left
    checked       who was asked (the crew trace) in plain words
    alternatives  up to 4 questions this drawing can answer, closest to the asked one first, built from the
                  drawing's own room labels, storeys and sheets so they work when clicked

No LLM: a refusal should be instant and free.
"""
from __future__ import annotations

import re

from .core import Answer

CARRIES = {
    "swiss_dwellings": "This is a 2D CAD floor plan. It holds room outlines with labels and areas, apartments, doors, windows and "
                       "sanitary fixtures, but no heights, materials, construction years or title-block details.",
    "ifc_models": "This is a BIM model. It holds storeys, spaces, building elements and whatever properties the author filled in; "
                  "anything not modelled or left empty cannot be read from it.",
    "ifc_bench": "This is a BIM model. It holds storeys, spaces, building elements and whatever properties the author filled in; "
                 "anything not modelled or left empty cannot be read from it.",
    "cubicasa5k": "This is a scanned floor plan. Detectors count rooms, doors, windows and fixtures on it; anything else has to be "
                  "read from the image, and these plans carry no scale bar, north arrow or title block.",
    "opening_design": "This is a PDF drawing sheet. Its printed text (schedules, notes, title block) and the image are what can be read.",
}
SCAN = "This is a scanned drawing. Only the image can be read, so answers are estimates and anything not drawn or written on it is out of reach."
BOILERPLATE = re.compile(r"^(?:Not answerable from this drawing's data:\s*|AI vision estimate \([^)]*\)\s*·\s*|The sheet's text does not contain the answer\s*·\s*)", re.I)
STOP = {"what", "which", "where", "when", "does", "this", "that", "there", "many", "much", "drawing", "sheet", "plan", "show", "shown",
        "drawn", "with", "from", "have", "they", "their", "each", "total", "number", "the", "are", "how", "is", "of", "in", "on", "a", "an"}


def _words(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-zäöüéèàß]{3,}", text.lower()):
        if w not in STOP:
            out.add(w[:-1] if w.endswith("s") and len(w) > 4 else w)
    return out


def category(answer: Answer, trace: list[dict], llm_ok: bool) -> str:
    if answer.intent == "scoped":  # stoa_qa.scope: the question names a room / apartment we could not pin down
        return "ambiguous"
    notes = " ".join([answer.method or ""] + [t.get("note") or "" for t in trace]).lower()
    # the planner's decline contract (stoa_qa.planner.answer): unanswerable / error / unknown / plan_* with no rows
    if re.search(r"fea\w*_api_key is not set|api key", notes):
        return "needs_llm"
    data_gap = (answer.intent == "unanswerable" or re.search(
        r"not answerable|does not contain|no rows match|is not recorded in this drawing|no numeric values found|"
        r"not (?:present|available|recorded|in the|visible|shown|indicated|specified|given|listed|provided)|"
        r"no .* (?:found|stored|visible|shown|indicated)|none in this model|cannot be (?:read|determined)|"
        r"does not (?:show|include|indicate|specify)", notes))
    if not data_gap and re.search(r"planner produced an invalid query|none of the planned queries", notes):
        return "not_understood"  # the question could not be turned into a query over this drawing's data
    if not data_gap and re.search(r"bad reply|timed out|timeout|http \d|unavailable:|error", notes):
        return "failed"
    if data_gap:
        return "not_in_data"
    if not llm_ok and not any(t.get("outcome") == "answered" for t in trace):
        return "needs_llm"
    return "not_understood"


HEADLINE = {
    "not_in_data": "This drawing doesn't contain that information",
    "not_understood": "I couldn't match this question to what the drawing contains",
    "needs_llm": "Free-form questions need the language model, which isn't set up",
    "failed": "A reader failed on this question",
    "ambiguous": "Which room or apartment do you mean?",
}
NEXT = {
    "not_in_data": "Ask about something the drawing does hold, or check another sheet of the same project.",
    "not_understood": "Try one of the questions below, or name the room, storey or element the way the drawing labels it.",
    "needs_llm": "Add FEATHERLESS_API_KEY to .env for free-form questions; the questions below work without it.",
    "failed": "Asking again often works; the questions below use other readers.",
    "ambiguous": "Name it the way the drawing labels it (a room number like 2.14, a label, or an apartment letter).",
}


def _specific_note(answer: Answer, trace: list[dict]) -> str:
    """The most informative thing an engine said, minus boilerplate."""
    cands = [answer.method or ""] + [t.get("note") or "" for t in reversed(trace)]
    for c in cands:
        c = BOILERPLATE.sub("", c.strip())
        if len(c) > 25 and not re.search(r"question not understood|no solver matched|name the room", c, re.I):
            c = c[:260].rstrip(" .")
            return c[0].upper() + c[1:] + "."
    return ""


def _capabilities(d) -> list[str]:
    """Questions that work on this drawing, filled with its own names."""
    from . import catalog, graph

    src = d.source
    out: list[str] = []
    try:
        g = graph.for_path(d.ask_path) if src in ("swiss_dwellings", "ifc_models", "ifc_bench") else None
    except Exception:
        g = None
    if g is not None:
        rooms = g.rooms()
        corridor = next((r for r in rooms if re.search(r"korridor|corridor|couloir|corridoio|flur|gang|hall", r.label, re.I)), None)
        stair = next((r for r in rooms if graph.STAIR.search(r.label)), None)
        kitchen = next((r for r in rooms if re.search(r"küche|kitchen|cuisine|cucina", r.label, re.I)), None)
        big = max((r for r in rooms if r.area), key=lambda r: r.area, default=None)
        if corridor:
            out += [f"Which rooms open onto {corridor.name}?", f"How wide is {corridor.name}?"]
        if kitchen and stair:
            out.append(f"How do I get from {kitchen.name} to {stair.name}?")
        elif rooms:
            out.append(f"How do I get from {rooms[0].name} to the exit?")
        if big:
            out.append(f"Which rooms are next to {big.name}?")
        out.append("Which rooms have no window?")
    if src == "swiss_dwellings":
        out += ["Which room is the largest?", "How many apartments are shown on this floor plan?", "How many doors are drawn on this floor plan?",
                "Which rooms have a toilet?", "What is the total net floor area of all rooms?"]
    elif src in ("ifc_models", "ifc_bench"):
        out += ["Which room is the largest?", "How many windows are there on each storey?", "Which software created this model?",
                "What is the total length of the exterior walls?", "What are the heights of the storeys?"]
        try:
            from . import mep

            if mep.for_path(d.ask_path) is not None:
                out += ["Which component types make up the plumbing system?", "What is the total length of pipe by storey?"]
        except Exception:
            pass
    elif src == "cubicasa5k":
        out += ["How many bedrooms are there?", "How many bathrooms are there?", "How many doors are shown on this floor plan?",
                "How many windows are shown on this floor plan?", "Is there a sauna?"]
    elif src == "opening_design":
        out += ["Which other sheets does this sheet refer to?", "What is the title of this sheet?", "Who is the architect on the title block?",
                "What is the scale of this drawing?"]
    else:
        out += ["What does this drawing show?", "What text is written on the drawing?", "When was this drawing made?"]
    stoa = [q["question"] for q in catalog.questions_for(d) if re.match(r"(?:How|What|Which|Is|Are|Does)\b", q["question"])][:30]  # English ones
    return list(dict.fromkeys(out + stoa))


def _per_candidate(question: str, method: str) -> list[str]:
    """'How many windows are in the kitchen?' + 'rooms match: 2.02 Küche, 2.15 Küche' -> one question per room number."""
    from .graph import QUESTION_WORDS

    nums = list(dict.fromkeys(re.findall(r"(?<![\w.])(U?\d+\.\d+|[A-Z]\d{3,4}(?:-[A-Z])?)(?![\w.])", method)))[:4]
    if not nums:
        return []
    span = None
    for pat in QUESTION_WORDS.values():
        m = re.search(r"(?:\bthe |\ba |\ban )?(?:" + pat + r")\w*", question, re.I)
        if m:
            span = m.span()
            break
    out = []
    for n in nums:
        if span:
            out.append(question[:span[0]] + f"room {n}" + question[span[1]:])
        else:
            out.append(question.rstrip(" ?") + f" (room {n})?")
    return out


def explain(d, question: str, answer: Answer, trace: list[dict], llm_ok: bool) -> dict | None:
    if answer.value is not None:
        return None
    cat = category(answer, trace, llm_ok)
    carries = CARRIES.get(d.source, SCAN)
    note = _specific_note(answer, trace)
    checked = [f"{t['agent']}: {('answered' if t['outcome'] == 'answered' else 'could not answer')}"
               + (f" — {BOILERPLATE.sub('', t['note'])[:120]}" if t.get("note") and t["outcome"] != "answered" else "")
               for t in trace if t.get("agent") != "Auditor"]
    if cat == "ambiguous":  # offer the same question once per candidate room / apartment the engine listed
        alts = _per_candidate(question, answer.method or "")
        if alts:
            return {"category": cat, "headline": HEADLINE[cat], "detail": note or carries, "next": NEXT[cat],
                    "checked": checked, "alternatives": alts}
    asked = _words(question)
    caps = _capabilities(d)
    ranked = sorted(caps, key=lambda c: (-len(asked & _words(c)), caps.index(c)))
    alts = [c for c in ranked if c.strip().lower() != question.strip().lower()][:4]
    return {"category": cat, "headline": HEADLINE[cat], "detail": " ".join(x for x in (note, carries) if x),
            "next": NEXT[cat], "checked": checked, "alternatives": alts}
