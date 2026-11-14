"""The auditor: checks an answer against evidence the answering engine did not use, before it is shown.

Each check is independent of how the answer was computed:

    printed areas    Swiss sheets print each room's area as text; the answer comes from polygon geometry
    printed labels   how often a room label is printed vs how many rooms were counted with it
    stated vs shape  IFC NetFloorArea quantity vs the area of the space's own geometry
    second reader    raster plans: do independent readers (network, detector, Gemma, Gemini) agree?
    marked items     a count must have one marked item per counted thing
    answer shape     an area question answered in m², a count answered with a whole number
    scope            a floor/storey/model-wide answer to a question about one apartment or room
    sheet notes      PDF sheets: general notes / legend lines that mention what was asked (shown, not scored)

A failed check lowers the confidence shown on the card; an independent confirmation raises it.
No LLM, no network: every check is computed.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .core import Answer, source_of
from .lang import quantity_of

TOL = 0.02  # Stoa's tolerance for areas, lengths and volumes
STOP = {"what", "which", "where", "when", "does", "this", "that", "there", "many", "much", "drawing", "sheet", "plan",
        "floor", "show", "shown", "drawn", "with", "from", "have", "they", "their", "each", "total", "number", "listed"}


@dataclass
class Check:
    name: str
    status: str        # pass | warn | fail | info
    detail: str
    independent: bool = True  # an independent source confirming the answer (raises confidence when it passes)

    def to_dict(self) -> dict:
        return asdict(self)


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol * max(abs(b), 1e-9)


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


# ------------------------------------------------------------------ checks

def answer_shape(question: str, a: Answer) -> list[Check]:
    v = _num(a.value)
    if v is None or a.unit is None:
        return []
    asked = quantity_of(question)
    want = {"area": ("m2", "ft2"), "volume": ("m3",), "length": ("m",), "count": ("count",)}[asked]
    if a.unit not in want:
        return [Check("answer shape", "fail", f"The question asks for {asked}, the answer is in {a.unit}.", False)]
    if a.unit == "count" and v != int(v):
        return [Check("answer shape", "fail", f"A count must be a whole number, got {v}.", False)]
    return []


def marked_items(a: Answer, engine: str) -> list[Check]:
    v = _num(a.value)
    if v is None or a.unit != "count" or engine in ("vision", "planner"):
        return []
    located = [e for e in a.evidence if e.coords or e.kind == "ifc"]
    if not located:
        return [Check("marked items", "warn", f"{int(v)} counted, but none of them could be marked on the drawing.", False)]
    if len(located) == int(v):
        return [Check("marked items", "pass", f"Each of the {int(v)} counted items is marked; step through them to check.", False)]
    return [Check("marked items", "warn", f"{int(v)} counted, {len(located)} marked: check the difference by eye.", False)]


def swiss_printed(path: str, question: str, a: Answer) -> list[Check]:
    from shapely.geometry import Point

    from . import swiss

    fl = swiss.load_floor(Path(path).stem)
    out: list[Check] = []
    v = _num(a.value)
    rooms = {r.number: r for r in fl.numbered}
    ids = [str(e.viewer_id) for e in a.evidence if e.viewer_id]
    if v is not None and a.unit == "m2":
        picked = {rooms[i.split(":", 1)[1]].handle: rooms[i.split(":", 1)[1]] for i in ids
                  if i.startswith("room:") and i.split(":", 1)[1] in rooms}
        by_rooms = bool(picked)
        if not picked:  # an apartment: its indoor rooms, as the area answer sums them
            for i in ids:
                if i.startswith("unit:"):
                    unit = next((u for h, u in fl.units if h == i.split(":", 1)[1]), None)
                    if unit is not None:
                        picked.update({r.handle: r for r in fl.numbered if r.anchor and unit.contains(Point(r.anchor))
                                       and r.indoor_net and (r.label or "").lower() not in swiss.STAIRWELL})
        if picked:
            with_text = [r for r in picked.values() if r.printed_area is not None]
            printed, computed = sum(r.printed_area for r in with_text), sum(r.area for r in with_text)
            scope = "" if by_rooms and len(picked) == 1 else " This confirms the room areas, not which rooms belong in the total."
            if len(with_text) == len(picked):
                out.append(Check("printed areas", "pass" if _close(v, printed) else "fail",
                                 f"The areas printed on the sheet for {len(picked)} room(s) add up to {printed:.2f} m²; the geometry gives {v:.2f} m².{scope}",
                                 independent=not scope))
            elif with_text:  # compare the rooms that do print an area; the rest (shafts, small rooms) carry no text
                ok = _close(computed, printed)
                out.append(Check("printed areas", "pass" if ok else "fail",
                                 f"{len(with_text)} of {len(picked)} rooms print their area: {printed:.2f} m² printed vs {computed:.2f} m² "
                                 f"from geometry for those rooms; the other {len(picked) - len(with_text)} have no printed area.{scope}",
                                 independent=False))
    if v is not None and a.unit == "count" and ids and all(i.startswith("room:") for i in ids):
        labels = {rooms[i.split(':', 1)[1]].label for i in ids if i.split(":", 1)[1] in rooms}
        if len(labels) == 1 and None not in labels:
            lab = labels.pop()
            printed = sum(t == lab for t, *_ in fl.room_texts)
            out.append(Check("printed labels", "pass" if printed == int(v) else "warn",
                             f"The label '{lab}' is printed {printed} time(s) on the sheet; {int(v)} room(s) were counted."))
    return out


def ifc_stated_vs_shape(path: str, a: Answer, limit: int = 40) -> list[Check]:
    if a.unit != "m2" or _num(a.value) is None:
        return []
    import ifcopenshell.util.shape as ushape

    from . import ifc

    f = ifc.ifc_path_for(path)
    if not f.exists():
        return []
    m = ifc.load_model(str(f))
    spaces = []
    for e in a.evidence[:limit]:
        if e.kind == "ifc" and e.ref:
            try:
                el = m.f.by_guid(e.ref)
            except RuntimeError:
                continue
            if el.is_a("IfcSpace"):
                spaces.append(el)
    pairs = []
    for sp in spaces:
        stated = m.qto(sp, "NetFloorArea")
        g = m.shape(sp)
        shape = ushape.get_footprint_area(g) if g is not None else None
        if stated and shape:
            pairs.append((sp, float(stated), float(shape)))
    if not pairs:
        return []
    off = [(sp, s, g) for sp, s, g in pairs if not _close(s, g, 0.05)]
    if not off:
        return [Check("stated vs shape", "pass", f"For {len(pairs)} space(s) the model's NetFloorArea matches the area of its geometry (within 5%).")]
    sp, s, g = off[0]
    return [Check("stated vs shape", "warn", f"{len(off)} of {len(pairs)} space(s) differ: '{sp.LongName or sp.Name}' states {s:.2f} m² "
                                             f"but its geometry measures {g:.2f} m² (finishes, columns or an outdated quantity).")]


def second_reader(path: str, a: Answer) -> list[Check]:
    from . import cubicasa

    v = _num(a.value)
    if v is None or not a.intent:
        return []
    preds = cubicasa.predictions_for(path)
    votes = {s: int(p["counts"][a.intent]) for s, p in preds.items() if a.intent in p.get("counts", {})}
    if len(votes) < 2:
        return [Check("second reader", "warn", "Only one reader has counted this plan, so nothing confirms it.", False)]
    agree = [s for s, x in votes.items() if x == int(v)]
    others = ", ".join(f"{cubicasa.READER[s]} {x}" for s, x in votes.items())
    if len(agree) >= 2:
        return [Check("second reader", "pass", f"{len(agree)} of {len(votes)} independent readers count {int(v)} ({others}).")]
    return [Check("second reader", "warn", f"No other reader agrees ({others}); check the marks on the plan.")]


def sheet_notes(d, question: str) -> list[Check]:
    """Lines of general notes / legends on a PDF sheet that mention what the question asks about."""
    import pymupdf

    from . import sheets

    pdf = next((s for s in d.sheets if s.kind == "pdf"), None)
    if pdf is None:
        return []
    try:
        doc = pymupdf.open(sheets.safe_path(pdf.path))
        text = doc[pdf.page].get_text()
    except Exception:
        return []
    if len(text.strip()) < 40:
        return []
    words = {w for w in re.findall(r"[a-zäöüéè]{4,}", question.lower()) if w not in STOP}
    words |= {w[:-1] for w in words if w.endswith("s")}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    notes, in_notes = [], False
    for ln in lines:
        up = ln.upper()
        if re.search(r"GENERAL NOTES|SHEET NOTES|KEYNOTES|LEGEND|\bNOTES?\b:?$", up):
            in_notes = True
            continue
        if (in_notes or re.match(r"^(\d+\.|[A-Z]\.|NOTE:|SEE |TYP)", up)) and any(re.search(r"\b" + re.escape(w), ln.lower()) for w in words):
            notes.append(ln[:160])
        if len(notes) >= 3:
            break
    if not notes:
        return []
    return [Check("sheet notes", "info", "Notes on the sheet that mention this: " + " | ".join(f"“{n}”" for n in notes), False)]


# ------------------------------------------------------------------ the auditor

# answers that cover a whole floor, storey or model (the template engines' floor-wide intents)
WIDE = {"apartments", "doors", "windows", "basins", "bathtubs", "showers", "toilets", "floor_area", "total_space_area"}
# words for what such an answer counts, which are not its scope ("toilets (WC)" names no bathroom)
COUNTED = {"toilets": r"toilet\w*|\bwcs?\b", "stair_count": r"stair\w*|trepp\w*|escaler\w*|scala\w*",
           "bathtubs": r"bath\w*|badewann\w*|vasch\w* da bagno", "showers": r"shower\w*|dusch\w*"}


def scope_check(question: str, a: Answer) -> list[Check]:
    """A floor-, storey- or model-wide answer to a question that names one apartment or room is wrong by
    construction ("28 windows in the kitchen"). Independent of the engine: reads only the question."""
    from . import scope

    intent = a.intent or ""
    wide = intent in WIDE or re.fullmatch(r"(?:door|window|wall|column|beam|stair|slab|sanitary|light)_count", intent)
    if not wide:
        return []
    q = re.sub(COUNTED.get(a.intent, r"(?!)"), " ", question, flags=re.I)
    apt = scope.apartment(q) if a.intent != "apartments" else scope.apartment(q.replace("apartments", " "))
    if apt or scope.mentions_room(q):
        what = f"apartment {apt}" if apt else "one room"
        return [Check("scope", "fail", f"The question is about {what}, but the answer covers the whole drawing "
                                       f"({a.intent.replace('_', ' ')}).")]
    return []


def run(d, question: str, a: Answer, engine: str) -> tuple[list[Check], float]:
    """Checks for one answer and the confidence to show after them."""
    if a.value is None:
        return [], a.confidence
    checks: list[Check] = []
    src = source_of(d.ask_path)
    try:
        checks += answer_shape(question, a)
        checks += scope_check(question, a)
        checks += marked_items(a, engine)
        if src == "swiss_dwellings":
            checks += swiss_printed(d.ask_path, question, a)
        if src in ("ifc_models", "ifc_bench"):
            checks += ifc_stated_vs_shape(d.ask_path, a)
        if src == "cubicasa5k" and engine == "detection":
            checks += second_reader(d.ask_path, a)
        if src == "opening_design" or engine in ("document", "vision"):
            checks += sheet_notes(d, question)
    except Exception as e:  # an auditor bug must never hide the answer
        checks.append(Check("auditor", "info", f"A check could not run ({type(e).__name__}).", False))
    if engine == "vision":
        checks.append(Check("vision only", "warn", "Read from the image by a model; no geometry was available to confirm it.", False))
    if not checks:
        checks.append(Check("no second source", "info",
                            "Nothing independent of the source data was available to check this against; the evidence is listed for checking by eye.", False))
    conf = a.confidence
    if any(c.status == "fail" for c in checks):
        conf = min(conf, 0.4)
    elif any(c.status == "warn" for c in checks):
        conf = min(conf, 0.7)
    elif any(c.status == "pass" and c.independent for c in checks):
        conf = max(conf, 0.95)
    return checks, conf
