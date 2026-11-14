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
    anchor: tuple[float, float] | None = None
    texts: list[str] = field(default_factory=list)

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
    doc = ezdxf.readfile(DATASET / "data/swiss_dwellings" / f"{stem}.dxf")
    msp = doc.modelspace()
    outlines, units, texts, doors, windows = [], [], [], [], []
    fx_rects, fx_other = [], []
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
            elif layer == "DOOR" and len(e) == 5:
                p = _poly(e)
                if p is not None:
                    doors.append((e.dxf.handle, p))
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
            texts.append((e.dxf.text.strip(), (ins.x, ins.y)))

    # rooms: net area = gross - directly nested room outlines (shafts, voids inside rooms)
    rooms = [Room(handle=h, poly=p, gross=p.area) for h, p in outlines]
    contains = {i: [j for j, o in enumerate(rooms) if j != i and o.gross < r.gross and r.poly.buffer(1e-6).contains(o.poly)]
                for i, r in enumerate(rooms)}
    for i, r in enumerate(rooms):
        direct = [j for j in contains[i] if not any(j in contains[k] for k in contains[i])]
        r.area = r.gross - sum(rooms[j].gross for j in direct)

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
    return Floor(stem, rooms, units, apt_tags, doors, windows, fixtures)


# ---------------------------------------------------------------- intents

APT_Q = re.compile(r"(?:apartment|wohnung|appartement|appartamento)\s+([A-Z])\b", re.I)
ROOM_Q = re.compile(r"(?<![\w.])(U?\d+\.\d+)(?![\w.])")
COUNT_KEYS = [
    ("apartments", r"apartment|wohnung|appartement|appartament|unità abitative"),
    ("doors", r"\bdoors?\b|türen|\bportes?\b|\bporte\b"),
    ("windows", r"window|fenster|fenêtre|finestr"),
    ("basins", r"washbasin|lavabo|lavabi"),
    ("bathtubs", r"bathtub|badewann|baignoire|vasch"),
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

    if (m := APT_Q.search(question)) and qty == "area":
        letter = m.group(1).upper()
        tag = [p for l, p in fl.apt_tags if l == letter]
        unit = next(((h, u) for h, u in fl.units if tag and u.contains(Point(tag[0]))), None)
        if not unit:
            return Answer(None, "m2", "apartment_area", 0.0, f"apartment {letter} outline not found", source_file=src)
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
        if qty == "area":
            return Answer(round(sum(r.area for r in rs), 2), "m2", "label_area", 0.95 if rs else 0.3,
                          f"sum of {len(rs)} rooms labelled '{lab}'", room_ev(rs), src)
        return Answer(len(rs), "count", "label_count", 0.97 if rs else 0.6,
                      f"{len(rs)} numbered rooms labelled '{lab}'", room_ev(rs), src)

    if qty == "area":
        rs = [r for r in fl.numbered if r.indoor_net]
        return Answer(round(sum(r.area for r in rs), 2), "m2", "floor_area", 0.9,
                      f"sum of {len(rs)} numbered indoor rooms (outdoor rooms and unlabelled shafts excluded)", room_ev(rs), src)

    ql = question.lower()
    for key, pat in COUNT_KEYS:
        if re.search(pat, ql):
            if key == "apartments":
                return Answer(len(fl.apt_tags), "count", "apartments", 0.97,
                              f"{len(fl.apt_tags)} apartment tags ({', '.join(sorted(l for l, _ in fl.apt_tags))})",
                              [Evidence("text", f"apt-{l}", f"apartment {l}", None, [list(p)],
                                        next((f"unit:{h}" for h, u in fl.units if u.contains(Point(p))), None))
                               for l, p in fl.apt_tags], src)
            if key in ("doors", "windows"):
                items = fl.doors if key == "doors" else fl.windows
                return Answer(len(items), "count", key, 0.97, f"{len(items)} {key[:-1]} symbols on DXF layer {key[:-1].upper()}",
                              [Evidence("polygon", h, f"{key[:-1]} {i + 1}", None, _coords(p), f"{key[:-1]}:{h}")
                               for i, (h, p) in enumerate(items)], src)
            items = [f for f in fl.fixtures if FIXTURE_RULES[key](f)]
            return Answer(len(items), "count", key, 0.95, f"{len(items)} {key} symbols recognised on DXF layer FIXTURE",
                          [Evidence("polygon", f["handle"], f"{key[:-1]} {i + 1}", None, _coords(f["poly"]), f"fixture:{f['handle']}")
                           for i, f in enumerate(items)], src)

    return Answer(None, None, "unknown", 0.0, "swiss: question not understood", source_file=src)
