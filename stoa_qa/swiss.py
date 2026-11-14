"""swiss_dwellings: answer from the DXF that ships next to every PDF/PNG.

Everything is computed from vector geometry, so each counted item / summed
room carries its exact polygon as evidence.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import ezdxf
from shapely.geometry import LineString, Point, Polygon

from . import scope
from .core import DATASET, Answer, Evidence
from .lang import quantity_of

ROOM_NO = re.compile(r"^U?\d+\.\d+$")
APT_TAG = re.compile(r"^(?:Whg\.?|Wohnung|Appt\.?|App\.?|Apt\.?|Apartment|Flat)\s*([A-Z])$")
PRINTED_AREA = re.compile(r"m²$|m2$")

OUTDOOR = {
    "balkon", "balcon", "balcone", "balcony", "loggia", "terrasse", "terrazza", "terrace",
    "sitzplatz", "jardin d'hiver", "wintergarten", "giardino d'inverno", "winter garden",
    "veranda", "patio", "dachterrasse", "toit-terrasse", "lastrico solare", "roof terrace",
    "garten", "jardin", "giardino", "garden",
}
STAIRWELL = {"treppenhaus", "cage d'escalier", "vano scala", "staircase", "stairwell"}

FIXTURE_RULES = {
    "toilets": lambda s: s["ell"] == 1 and not s["circ"],
    "basins": lambda s: s["ell"] == 1 and len(s["circ"]) == 1,
    "bathtubs": lambda s: 69 in s["polys"],
    "showers": lambda s: 5 in s["polys"],
}


@dataclass
class Room:
    handle: str
    poly: Polygon
    gross: float
    area: float = 0.0
    number: str | None = None
    label: str | None = None
    printed_area: float | None = None
    printed_text: str | None = None
    anchor: tuple[float, float] | None = None
    texts: list[str] = field(default_factory=list)
    holes: list = field(default_factory=list)  # nested outlines (shafts, voids); subtracted from `area`

    @property
    def outdoor(self) -> bool:
        return (self.label or "").strip().lower() in OUTDOOR

    @property
    def indoor_net(self) -> bool:
        """Counts toward net floor area: labelled (unlabelled numbered = shaft/void) and not outdoor."""
        return not self.outdoor


@dataclass
class Floor:
    stem: str
    rooms: list[Room]
    units: list[tuple[str, Polygon]]          # (handle, outline)
    apt_tags: list[tuple[str, tuple[float, float]]]
    doors: list[tuple[str, Polygon]]
    windows: list[tuple[str, Polygon]]
    fixtures: list[dict]
    walls: list[tuple[str, Polygon]] = field(default_factory=list)
    door_lines: list = field(default_factory=list)  # 2-point opening lines on layer DOOR (the leaf swing is in `doors`)
    room_texts: list = field(default_factory=list)  # every ROOM_TEXT as (text, insert point, align point)

    @property
    def numbered(self) -> list[Room]:
        return [r for r in self.rooms if r.number]

    def labels(self) -> set[str]:
        return {r.label for r in self.rooms if r.label}


def _coords(poly: Polygon) -> list[list[float]]:
    return [[round(x, 4), round(y, 4)] for x, y in poly.exterior.coords]


def _poly(e) -> Polygon | None:
    pts = [(p[0], p[1]) for p in e.get_points("xy")]
    if len(pts) < 3:
        return None
    p = Polygon(pts)
    return p if p.is_valid and p.area > 0 else p.buffer(0) if p.area > 0 else None


@lru_cache(maxsize=256)
def load_floor(stem: str) -> Floor:
    """The floor whose DXF is ``data/swiss_dwellings/<stem>.dxf`` inside the dataset."""
    return read_dxf(str((DATASET / "data/swiss_dwellings" / f"{stem}.dxf").resolve()))


@lru_cache(maxsize=256)
def read_dxf(path: str) -> Floor:
    """Parse one floor-plan DXF. This is the only DXF reader: answers and the 3D scene both call it."""
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    outlines, units, texts, doors, windows, walls, door_lines = [], [], [], [], [], [], []
    fx_rects, fx_other, room_texts = [], [], []
    for e in msp:
        layer, kind = e.dxf.layer, e.dxftype()
        if kind == "LWPOLYLINE":
            if layer == "ROOM_OUTLINE":
                p = _poly(e)
                if p is not None:
                    outlines.append((e.dxf.handle, p))
            elif layer == "UNIT_OUTLINE":
                p = _poly(e)
                if p is not None:
                    units.append((e.dxf.handle, p))
            elif layer == "WALL":
                p = _poly(e)
                if p is not None:
                    walls.append((e.dxf.handle, p))
            elif layer == "DOOR" and len(e) == 5:
                p = _poly(e)
                if p is not None:
                    doors.append((e.dxf.handle, p))
            elif layer == "DOOR" and len(e) == 2:
                door_lines.append([(p[0], p[1]) for p in e.get_points("xy")])
            elif layer == "WINDOW" and e.closed:
                p = _poly(e)
                if p is not None:
                    windows.append((e.dxf.handle, p))
            elif layer == "FIXTURE":
                pts = [(p[0], p[1]) for p in e.get_points("xy")]
                if e.closed and len(pts) == 4:
                    fx_rects.append((e.dxf.handle, Polygon(pts)))
                elif len(pts) > 1:
                    fx_other.append(("poly", len(pts), LineString(pts)))
        elif kind == "CIRCLE" and layer == "FIXTURE":
            c = e.dxf.center
            fx_other.append(("circle", e.dxf.radius, Point(c.x, c.y).buffer(e.dxf.radius)))
        elif kind == "TEXT" and layer == "ROOM_TEXT":
            ins = e.dxf.insert
            al = e.dxf.align_point if e.dxf.hasattr("align_point") else ins
            texts.append((e.dxf.text.strip(), (ins.x, ins.y)))
            room_texts.append((e.dxf.text.strip(), (ins.x, ins.y), (al.x, al.y)))

    # rooms: net area = gross - directly nested room outlines (shafts, voids inside rooms)
    rooms = [Room(handle=h, poly=p, gross=p.area) for h, p in outlines]
    contains = {i: [j for j, o in enumerate(rooms) if j != i and o.gross < r.gross and r.poly.buffer(1e-6).contains(o.poly)]
                for i, r in enumerate(rooms)}
    for i, r in enumerate(rooms):
        direct = [j for j in contains[i] if not any(j in contains[k] for k in contains[i])]
        r.area = r.gross - sum(rooms[j].gross for j in direct)
        r.holes = [rooms[j].poly for j in direct]

    apt_tags = []
    for t, pos in texts:
        m = APT_TAG.match(t)
        if m:
            apt_tags.append((m.group(1), pos))
            continue
        cands = [r for r in rooms if r.poly.contains(Point(pos))]
        if not cands:
            continue
        r = min(cands, key=lambda r: r.gross)
        r.texts.append(t)
        if ROOM_NO.match(t) and r.number is None:
            r.number, r.anchor = t, pos
        elif PRINTED_AREA.search(t):
            r.printed_text = t
            try:
                r.printed_area = float(t.split()[0].replace(",", "."))
            except ValueError:
                pass
        elif r.label is None:
            r.label = t

    fixtures = []
    for h, R in fx_rects:
        if any(q.buffer(0.01).contains(R) for _, q in fx_rects if q is not R):
            continue  # inner outline of another symbol
        B = R.buffer(0.01)
        inside = [o for o in fx_other if B.contains(o[2])]
        fixtures.append({
            "handle": h, "poly": R,
            "ell": sum(o[0] == "poly" and o[1] == 48 for o in inside),
            "circ": [o[1] for o in inside if o[0] == "circle"],
            "polys": sorted(o[1] for o in inside if o[0] == "poly" and o[1] not in (2, 48)),
        })
    return Floor(Path(path).stem, rooms, units, apt_tags, doors, windows, fixtures, walls, door_lines, room_texts)


# ---------------------------------------------------------------- intents

APT_Q = re.compile(r"(?:apartment|wohnung|appartement|appartamento)\s+([A-Z])\b", re.I)
ROOM_Q = re.compile(r"(?<![\w.])(U?\d+\.\d+)(?![\w.])")
COUNT_KEYS = [
    ("apartments", r"apartment|wohnung|appartement|appartament|unità abitative"),
    ("doors", r"\bdoors?\b|türen|\bportes?\b|\bporte\b"),
    ("windows", r"window|fenster|fenêtre|finestr"),
    ("basins", r"washbasin|lavabo|lavabi"),
    ("bathtubs", r"bathtub|badewann|baignoire|vasch\w* da bagno|vasch"),  # "vasche da bagno": the bagno is the tub, not a room
    ("showers", r"shower|dusche|douche|docc"),
    ("toilets", r"toilet|\bwcs?\b"),
]


def _label_in(question: str, floor: Floor) -> str | None:
    m = re.search(r"«([^»]+)»|„([^“”]+)[“”]", question) or re.search(r"(?:^|[\s(])'(.+)'(?=[\s)?,])", question)
    if m:
        return next(g for g in m.groups() if g).strip()
    hits = [l for l in floor.labels() if l in question]
    return max(hits, key=len) if hits else None


def _has_label_quotes(q: str) -> bool:
    return bool(re.search(r"«|„|(?:^|[\s(])'", q))


def answer(path: str, question: str) -> Answer:
    stem = Path(path).stem
    fl = load_floor(stem)
    qty = quantity_of(question)
    src = f"data/swiss_dwellings/{stem}.dxf"

    def room_ev(rs):
        return [Evidence("polygon", r.handle, f"{r.number or ''} {r.label or ''}".strip(), round(r.area, 3), _coords(r.poly),
                         f"room:{r.number or r.handle}") for r in rs]

    # Scope beyond the floor: "apartment B", or one room ("the kitchen", "room 2.03"). Never answer those with
    # the floor-wide number: count inside the scope, or decline so a later engine can try (stoa_qa/scope.py).
    apt = scope.apartment(question)
    unit = _unit(fl, apt) if apt else None

    def decline(why: str, intent: str = "scoped") -> Answer:
        return Answer(None, None, intent, 0.0, f"swiss: {why}", source_file=src)

    def one_room(ref, what: str):
        """The single Room a room reference means, or a decline naming the candidates."""
        if len(ref) != 1:
            return decline(f"{len(ref)} rooms match{' ' + what if what else ''}: {scope.which(ref)}. Name one, e.g. by its room number.")
        return next((r for r in fl.numbered if r.number == ref[0].number), None) or decline(f"room {ref[0].name} not found")

    def rooms_total(ref, where: str) -> Answer:
        """Sum over every room a reference matches ("all kitchens")."""
        nums = {n.number for n in ref}
        rs = [r for r in fl.numbered if r.number in nums]
        return Answer(round(sum(r.area for r in rs), 2), "m2", "rooms_area", 0.9,
                      f"sum of {len(rs)} rooms{where}: {', '.join(f'{r.number} {r.label}' for r in rs)} (DXF ROOM_OUTLINE)",
                      room_ev(rs), src)

    def living_area() -> Answer:
        """Living area = the apartments' indoor rooms (the apartment-area rule, per apartment), summed."""
        parts, rs_all, ev = [], [], []
        for letter in sorted({l for l, _ in fl.apt_tags}):
            u = _unit(fl, letter)
            if not u:
                continue
            rs = [r for r in fl.numbered if r.indoor_net and (r.label or "").lower() not in STAIRWELL
                  and r.anchor and u[1].contains(Point(r.anchor))]
            parts.append(f"{letter} {sum(r.area for r in rs):.2f}")
            rs_all += rs
            ev.append(Evidence("polygon", u[0], f"apartment {letter}", None, _coords(u[1]), f"unit:{u[0]}"))
        if not parts:
            return decline("no apartment outlines on this floor, so its living area is not defined", "living_area")
        return Answer(round(sum(r.area for r in rs_all), 2), "m2", "living_area", 0.85,
                      f"living area = the apartments' indoor rooms, summed (apartment {', '.join(parts)} m²); rooms outside "
                      f"any apartment (common stairwell, corridors) and outdoor spaces excluded", ev + room_ev(rs_all), src)

    if apt and qty == "area":
        if not unit:
            return Answer(None, "m2", "apartment_area", 0.0, f"apartment {apt} outline not found", source_file=src)
        named = scope.rooms_named(path, question, group=apt)
        if named and len(named[0]) > 1 and scope.wants_total(question):  # "all bedrooms of apartment B"
            return rooms_total(named[0], f" in apartment {apt}")
        if named:  # "the kitchen in apartment B": that room, not the whole apartment
            r = one_room(named[0], f"in apartment {apt}")
            if isinstance(r, Answer):
                return r
            return Answer(round(r.area, 2), "m2", "room_area", 0.95,
                          f"polygon area of room {r.number} ({r.label}) in apartment {apt}, minus nested shafts (DXF ROOM_OUTLINE)",
                          room_ev([r]), src)
        letter = apt
        rs = [r for r in fl.numbered if r.indoor_net and (r.label or "").lower() not in STAIRWELL and unit[1].contains(Point(r.anchor))]
        return Answer(round(sum(r.area for r in rs), 2), "m2", "apartment_area", 0.95,
                      f"sum of {len(rs)} indoor rooms inside apartment {letter} outline (DXF UNIT_OUTLINE)",
                      [Evidence("polygon", unit[0], f"apartment {letter}", None, _coords(unit[1]), f"unit:{unit[0]}")] + room_ev(rs), src)

    if (m := ROOM_Q.search(question)) and qty == "area":
        rs = [r for r in fl.rooms if r.number == m.group(1)]
        if not rs:
            return Answer(None, "m2", "room_area", 0.0, f"room {m.group(1)} not found", source_file=src)
        r = rs[0]
        return Answer(round(r.area, 2), "m2", "room_area", 0.97,
                      f"polygon area of room {r.number} minus nested shafts (DXF ROOM_OUTLINE)", room_ev([r]), src)

    if _has_label_quotes(question):
        lab = _label_in(question, fl)
        rs = [r for r in fl.numbered if r.label == lab]
        where = ""
        if apt:  # "rooms labelled 'Zimmer' in apartment B"
            if not unit:
                return decline(f"apartment {apt} outline not found")
            rs = [r for r in rs if r.anchor and unit[1].contains(Point(r.anchor))]
            where = f" in apartment {apt}"
        if qty == "area":
            return Answer(round(sum(r.area for r in rs), 2), "m2", "label_area", 0.95 if rs else 0.3,
                          f"sum of {len(rs)} rooms labelled '{lab}'{where}", room_ev(rs), src)
        return Answer(len(rs), "count", "label_count", 0.97 if rs else 0.6,
                      f"{len(rs)} numbered rooms labelled '{lab}'{where}", room_ev(rs), src)

    if qty == "area":
        if scope.living(question):  # "the total living area", "die Wohnfläche": the apartments, not the living room
            return living_area()
        named = scope.rooms_named(path, question)
        if named and len(named[0]) > 1 and scope.wants_total(question):  # "the total area of all kitchens"
            return rooms_total(named[0], "")
        if named:  # "the area of the kitchen": one room, not the whole floor
            r = one_room(named[0], "")
            if isinstance(r, Answer):
                return r
            return Answer(round(r.area, 2), "m2", "room_area", 0.95,
                          f"polygon area of room {r.number} ({r.label}) minus nested shafts (DXF ROOM_OUTLINE)", room_ev([r]), src)
        if scope.mentions_room(question):  # "the area of the living room" on a floor without one
            return decline("the question names a room this floor has no label for, so its area cannot be given")
        rs = [r for r in fl.numbered if r.indoor_net]
        return Answer(round(sum(r.area for r in rs), 2), "m2", "floor_area", 0.9,
                      f"sum of {len(rs)} numbered indoor rooms (outdoor rooms and unlabelled shafts excluded)", room_ev(rs), src)

    ql = question.lower()
    for key, pat in COUNT_KEYS:
        m = re.search(pat, ql)
        if not m:
            continue
        end = m.end() + len(re.match(r"\w*", ql[m.end():]).group(0))  # the whole word: "toilets", "Badewannen"
        rest = question[:m.start()] + " " + question[end:]  # the question without the words for what is counted
        if key == "apartments":
            if apt:
                continue  # "apartment B" says where to count, not what
            if scope.rooms_named(path, rest) or scope.mentions_room(rest):
                return decline("counting apartments by the rooms they contain is not a template computed here")
            return Answer(len(fl.apt_tags), "count", "apartments", 0.97,
                          f"{len(fl.apt_tags)} apartment tags ({', '.join(sorted(l for l, _ in fl.apt_tags))})",
                          [Evidence("text", f"apt-{l}", f"apartment {l}", None, [list(p)],
                                    next((f"unit:{h}" for h, u in fl.units if u.contains(Point(p))), None))
                           for l, p in fl.apt_tags], src)
        noun = key[:-1]
        if key in ("doors", "windows"):
            items = [(h, p) for h, p in (fl.doors if key == "doors" else fl.windows)]
            layer, kind = f"DXF layer {noun.upper()}", noun
        else:
            items = [(f["handle"], f["poly"]) for f in fl.fixtures if FIXTURE_RULES[key](f)]
            layer, kind = "DXF layer FIXTURE", "fixture"

        def ev(sel):
            return [Evidence("polygon", h, f"{noun} {i + 1}", None, _coords(p), f"{kind}:{h}") for i, (h, p) in enumerate(sel)]

        if apt and not unit:
            return decline(f"apartment {apt} outline not found")
        named = scope.rooms_named(path, rest, group=apt)
        if named:  # "windows in the kitchen", "doors of room 2.03"
            r = one_room(named[0], f"in apartment {apt}" if apt else "")
            if isinstance(r, Answer):
                return r
            sel, rule = _in_room(path, fl, r, key, items)
            return Answer(len(sel), "count", f"{key}_in_room", 0.9,
                          f"{len(sel)} {key} of room {r.number} ({r.label}): {rule}", ev(sel), src)
        if scope.mentions_room(rest):  # a room word ("the bathroom") that matches no room on this floor
            return decline("the question names a room this floor has no label for, so it cannot be counted inside it")
        if apt:  # "doors in apartment B"
            sel, rule = _in_apartment(path, fl, unit[1], apt, key, items)
            return Answer(len(sel), "count", f"{key}_in_apartment", 0.9, f"{len(sel)} {key} of apartment {apt}: {rule}", ev(sel), src)
        conf = 0.97 if key in ("doors", "windows") else 0.95
        what = f"{noun} symbols on {layer}" if key in ("doors", "windows") else f"{key} symbols recognised on {layer}"
        return Answer(len(items), "count", key, conf, f"{len(items)} {what}", ev(items), src)

    if apt or scope.rooms_named(path, question):
        return decline("this count inside an apartment or room is not a template computed from the DXF")
    return Answer(None, None, "unknown", 0.0, "swiss: question not understood", source_file=src)


def _unit(fl: Floor, letter: str) -> tuple[str, Polygon] | None:
    """(handle, outline) of the apartment whose tag is `letter`."""
    tag = [p for l, p in fl.apt_tags if l == letter]
    return next(((h, u) for h, u in fl.units if tag and u.contains(Point(tag[0]))), None)


def _in_apartment(path: str, fl: Floor, outline: Polygon, letter: str, key: str, items: list) -> tuple[list, str]:
    """The items of one apartment: those belonging to one of its rooms (doors and windows sit in walls, outside
    the room and apartment outlines, so they are assigned through their rooms like in the room graph)."""
    from . import graph

    if key == "doors":
        g = graph.for_path(path)
        hs = {e.via.ref for e in g.edges if e.kind == "door" and e.via is not None
              and any(g.nodes[n].group == letter for n in (e.a, e.b) if n in g.nodes)}
        return [(h, p) for h, p in items if h in hs], "doors leading into one of its rooms (room graph), entrance door included"
    rooms = [r for r in fl.numbered if r.anchor and outline.contains(Point(r.anchor))]
    if key == "windows":
        sel = [(h, p) for h, p in items if _window_room(fl, p) in rooms]
        return sel, "windows whose surroundings (0.3 m) overlap one of its rooms most"
    sel = [(h, p) for h, p in items if outline.contains(p.centroid)]
    return sel, "symbols with their centre inside its outline (DXF UNIT_OUTLINE)"


def _window_room(fl: Floor, p: Polygon) -> Room | None:
    """The room a window belongs to: the one overlapping its 0.3 m surroundings most (the room graph's rule)."""
    from . import graph

    reach = p.buffer(graph.DOOR_REACH)
    best = max(fl.numbered, key=lambda r: reach.intersection(r.poly).area, default=None)
    return best if best is not None and reach.intersects(best.poly) else None


def _in_room(path: str, fl: Floor, room: Room, key: str, items: list) -> tuple[list, str]:
    """The items of one room, with the rule used (the same rules the room graph uses)."""
    from . import graph

    if key == "doors":  # doors the room graph links to this room
        g = graph.for_path(path)
        hs = {e.via.ref for _, e in g.links(f"room:{room.number}", ("door",)) if e.via is not None}
        return [(h, p) for h, p in items if h in hs], "doors leading into it (room graph: door leaf within 0.3 m of the room)"
    if key == "windows":  # the room whose outline overlaps the window's surroundings most
        return [(h, p) for h, p in items if _window_room(fl, p) is room], "windows whose surroundings (0.3 m) overlap this room most"
    def inside(p):  # fixtures: the smallest room containing the symbol's centre
        c = [r for r in fl.numbered if r.poly.contains(p.centroid)]
        return min(c, key=lambda r: r.gross) if c else None
    return [(h, p) for h, p in items if inside(p) is room], "symbols with their centre inside it"


# ---------------------------------------------------------------- fact tables (for the query planner)

def _long_side(poly: Polygon) -> float:
    xs = list(poly.minimum_rotated_rectangle.exterior.coords)
    return max(math.dist(xs[0], xs[1]), math.dist(xs[1], xs[2]))


def fixture_type(f: dict) -> str:
    return next((k for k, rule in FIXTURE_RULES.items() if rule(f)), "other")


@lru_cache(maxsize=128)
def facts(stem: str) -> dict:
    """Queryable tables for one floor. Every row keeps the Evidence that locates it on the drawing."""
    fl = load_floor(stem)
    apt_of_unit = {h: next((l for l, p in fl.apt_tags if u.contains(Point(p))), None) for h, u in fl.units}

    def unit_of(pt) -> str | None:
        return next((apt_of_unit[h] for h, u in fl.units if u.contains(Point(pt))), None)

    def room_of(pt) -> Room | None:
        c = [r for r in fl.numbered if r.poly.contains(Point(pt))]
        return min(c, key=lambda r: r.gross) if c else None

    fixtures = []
    per_room: dict[str, dict[str, int]] = {}
    for f in fl.fixtures:
        t = fixture_type(f)
        if t == "other":
            continue
        c = f["poly"].centroid
        rm = room_of((c.x, c.y))
        if rm is not None:
            per_room.setdefault(rm.handle, {}).setdefault(t, 0)
            per_room[rm.handle][t] += 1
        fixtures.append({"id": f"fixture:{f['handle']}", "type": t[:-1] if t.endswith("s") else t,
                         "room": rm.number if rm else None, "room_label": rm.label if rm else None,
                         "_ev": Evidence("polygon", f["handle"], t[:-1], None, _coords(f["poly"]), f"fixture:{f['handle']}")})

    rooms = []
    for r in fl.numbered:
        fx = per_room.get(r.handle, {})
        rooms.append({"number": r.number, "label": r.label or "(unlabelled)", "area_m2": round(r.area, 2),
                      "apartment": unit_of(r.anchor), "outdoor": r.outdoor,
                      "toilets": fx.get("toilets", 0), "washbasins": fx.get("basins", 0),
                      "bathtubs": fx.get("bathtubs", 0), "showers": fx.get("showers", 0),
                      "_ev": Evidence("polygon", r.handle, f"{r.number} {r.label or ''}".strip(), round(r.area, 2),
                                      _coords(r.poly), f"room:{r.number or r.handle}")})

    apartments = []
    for h, u in fl.units:
        letter = apt_of_unit[h]
        if letter is None:
            continue
        rs = [r for r in fl.numbered if u.contains(Point(r.anchor))]
        indoor = [r for r in rs if r.indoor_net and (r.label or "").lower() not in STAIRWELL]
        apartments.append({"apartment": letter, "area_m2": round(sum(r.area for r in indoor), 2), "rooms": len(indoor),
                           "outdoor_area_m2": round(sum(r.area for r in rs if r.outdoor), 2),
                           "_ev": Evidence("polygon", h, f"apartment {letter}", None, _coords(u), f"unit:{h}")})

    def openings(items, kind):
        out = []
        for i, (h, p) in enumerate(items):
            c = p.centroid
            out.append({"id": f"{kind}:{h}", "width_m": round(_long_side(p), 2), "apartment": unit_of((c.x, c.y)),
                        "_ev": Evidence("polygon", h, f"{kind} {i + 1}", round(_long_side(p), 2), _coords(p), f"{kind}:{h}")})
        return out

    def fx(r):
        have = [f"{r[k]} {k[:-1] if r[k] == 1 else k}" for k in ("toilets", "washbasins", "bathtubs", "showers") if r[k]]
        return f" · {', '.join(have)}" if have else ""

    return {
        "rooms": {"rows": rooms, "units": {"area_m2": "m2"},
                  "show": lambda r: f"{r['number']} {r['label']} — {r['area_m2']:.2f} m²" + (f" · apt {r['apartment']}" if r["apartment"] else "") + fx(r),
                  "doc": "numbered rooms: number, label (as printed), area_m2 (net), apartment (letter or null), "
                         "outdoor (balcony/terrace/loggia), toilets/washbasins/bathtubs/showers (fixtures inside)"},
        "apartments": {"rows": apartments, "units": {"area_m2": "m2", "outdoor_area_m2": "m2"},
                       "show": lambda r: f"Apartment {r['apartment']} — {r['area_m2']:.2f} m² indoor · {r['rooms']} rooms"
                                         + (f" · {r['outdoor_area_m2']:.2f} m² outdoor" if r["outdoor_area_m2"] else ""),
                       "doc": "residential units: apartment (letter), area_m2 (indoor net), rooms (count), outdoor_area_m2"},
        "doors": {"rows": openings(fl.doors, "door"), "units": {"width_m": "m"}, "doc": "doors: width_m, apartment",
                  "show": lambda r: f"{r['_ev'].label} — {r['width_m']:.2f} m wide" + (f" · apt {r['apartment']}" if r["apartment"] else "")},
        "windows": {"rows": openings(fl.windows, "window"), "units": {"width_m": "m"}, "doc": "windows: width_m, apartment",
                    "show": lambda r: f"{r['_ev'].label} — {r['width_m']:.2f} m wide" + (f" · apt {r['apartment']}" if r["apartment"] else "")},
        "fixtures": {"rows": fixtures, "units": {}, "doc": "sanitary fixtures: type (toilet|basin|bathtub|shower), room (number), room_label",
                     "show": lambda r: f"{r['type']} in room {r['room'] or '?'}" + (f" ({r['room_label']})" if r["room_label"] else "")},
    }
