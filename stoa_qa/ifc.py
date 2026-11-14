"""ifc_models: answer from the IFC model (also for the PDF plans rendered from it).

Quantities come from IFC quantity sets when the authoring tool exported them,
otherwise from tessellated geometry. Evidence = IFC GlobalIds, which the 3D
viewer highlights directly.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as uel
import ifcopenshell.util.shape as ushape
import ifcopenshell.util.unit as uunit

from .core import DATASET, Answer, Evidence
from .lang import is_model_scope, objects_in, quantity_of

CLASSES = {
    "door": ["IfcDoor"],
    "window": ["IfcWindow"],
    "space": ["IfcSpace"],
    "wall": ["IfcWall"],
    "column": ["IfcColumn"],
    "beam": ["IfcBeam"],
    "stair": ["IfcStair"],
    "slab": ["IfcSlab"],
    "sanitary": ["IfcSanitaryTerminal"],
    "light": ["IfcLightFixture"],
}
# IFC2X3 stores these as generic IfcFlowTerminal; the real kind is on the linked type object
FLOW_TYPES = {"sanitary": "IfcSanitaryTerminalType", "light": "IfcLightFixtureType"}
# priority when several object words appear ("net side area of all walls" mentions no room)
PRIORITY = ["wall", "slab", "door", "window", "column", "beam", "stair", "sanitary", "light", "space"]

_GEOM = ifcopenshell.geom.settings()
_GEOM.set("use-world-coords", True)


def ifc_path_for(path: str) -> Path:
    stem = Path(path).stem
    stem = re.sub(r"_plan-.*$", "", stem)
    return DATASET / "data/ifc_models" / f"{stem}.ifc"


class Model:
    def __init__(self, path: Path):
        self.path = path
        self.f = ifcopenshell.open(str(path))
        self.storeys = {s.Name: s for s in self.f.by_type("IfcBuildingStorey") if s.Name}
        self._geom: dict[int, object] = {}
        self.length_scale = uunit.calculate_unit_scale(self.f)  # project length unit -> metres

    # ---- spatial helpers
    def storey_of(self, el):
        c = uel.get_container(el)
        if c is None:
            c = uel.get_aggregate(el)
        while c is not None and not c.is_a("IfcBuildingStorey"):
            c = uel.get_aggregate(c) if not c.is_a("IfcBuilding") else None
        return c

    def elements(self, cls: str, storey=None):
        out = []
        for t in CLASSES[cls]:
            try:
                out += self.f.by_type(t)
            except RuntimeError:
                pass  # class not in this schema
        if cls in FLOW_TYPES:
            for e in self.f.by_type("IfcFlowTerminal"):
                ty = uel.get_type(e)
                if e not in out and ty is not None and ty.is_a(FLOW_TYPES[cls]):
                    out.append(e)
        # count whole objects only: drop parts aggregated into a parent of the same class (stair -> risers)
        out = [e for e in out if not (e.Decomposes and e.Decomposes[0].RelatingObject.is_a(e.is_a()))]
        if storey is not None:
            out = [e for e in out if self.storey_of(e) == storey]
        return out

    # ---- quantities
    def qto(self, el, *names):
        for qs in uel.get_psets(el, qtos_only=True).values():
            for n in names:
                v = qs.get(n)
                if isinstance(v, (int, float)) and v:
                    return float(v)
        return None

    def shape(self, el):
        if el.id() not in self._geom:
            try:
                self._geom[el.id()] = ifcopenshell.geom.create_shape(_GEOM, el).geometry
            except Exception:
                self._geom[el.id()] = None
        return self._geom[el.id()]

    def space_area(self, sp):
        v = self.qto(sp, "NetFloorArea")
        if v:
            return v, "IFC quantity NetFloorArea"
        g = self.shape(sp)
        return (ushape.get_footprint_area(g), "footprint of IfcSpace geometry") if g else (None, "no geometry")

    def volume(self, el):
        v = self.qto(el, "NetVolume", "GrossVolume")
        if v:
            return v, "quantity"
        g = self.shape(el)
        return (ushape.get_volume(g), "geometry") if g else (0.0, "none")

    def wall_length(self, el):
        return (self.qto(el, "Length") or 0.0) * self.length_scale

    def wall_side_area(self, el):
        return self.qto(el, "NetSideArea", "GrossSideArea") or 0.0

    def space_keys(self, sp) -> list[str]:
        keys = []
        if sp.LongName and sp.Name:
            keys += [f"{sp.LongName} ({sp.Name})", f"{sp.Name} ({sp.LongName})"]
        keys += [k for k in (sp.LongName, sp.Name) if k]
        return keys


@lru_cache(maxsize=16)
def load_model(ifc: str) -> Model:
    return Model(Path(ifc))


def _longest_in(text: str, names) -> str | None:
    hits = [n for n in names if n and n in text]
    return max(hits, key=len) if hits else None


def _ev(m: Model, els, label_fn=lambda e: e.Name or e.is_a()):
    return [Evidence("ifc", e.GlobalId, label_fn(e)) for e in els]


def answer(path: str, question: str) -> Answer:
    ifc = ifc_path_for(path)
    if not ifc.exists():
        return Answer(None, None, "unknown", 0.0, f"IFC model {ifc.name} not found")
    m = load_model(str(ifc))
    src = f"data/ifc_models/{ifc.name}"
    qty = quantity_of(question)
    objs = objects_in(question)
    storey_name = None if is_model_scope(question) else _longest_in(question, m.storeys)
    storey = m.storeys.get(storey_name) if storey_name else None
    where = f"on storey '{storey_name}'" if storey is not None else "in the whole model"

    # --- single-room area: a known room name appears in the question
    if qty == "area" and "wall" not in objs:
        spaces = m.elements("space", storey)
        key_to_space = {}
        for sp in spaces:
            for k in m.space_keys(sp):
                key_to_space.setdefault(k, sp)
        key = _longest_in(question, key_to_space)
        if key and not re.search(r"\b(all|total|alle|samlede|totale|kokku|gesamte|łączna|todos)\b", question.lower()):
            sp = key_to_space[key]
            a, how = m.space_area(sp)
            return Answer(round(a, 2) if a else None, "m2", "room_area", 0.95 if a else 0.0,
                          f"room '{key}' {where}: {how}", [Evidence("ifc", sp.GlobalId, key, a)], src)
        tot, ev = 0.0, []
        for sp in spaces:
            a, _ = m.space_area(sp)
            if a:
                tot += a
                ev.append(Evidence("ifc", sp.GlobalId, sp.LongName or sp.Name or "space", round(a, 2)))
        return Answer(round(tot, 2), "m2", "total_space_area", 0.85, f"sum of {len(ev)} IfcSpace areas {where}", ev, src)

    cls = next((c for c in PRIORITY if c in objs), None)
    if cls is None:
        return Answer(None, None, "unknown", 0.0, "ifc: no element class recognised", source_file=src)
    els = m.elements(cls, storey)

    if qty == "volume":
        vols = [(e, m.volume(e)[0]) for e in els]
        return Answer(round(sum(v for _, v in vols), 2), "m3", f"{cls}_volume", 0.85,
                      f"sum of NetVolume (or mesh volume) of {len(els)} {CLASSES[cls][0]} {where}",
                      [Evidence("ifc", e.GlobalId, e.Name or cls, round(v, 3)) for e, v in vols], src)
    if qty == "area":
        ars = [(e, m.wall_side_area(e)) for e in els]
        return Answer(round(sum(a for _, a in ars), 2), "m2", f"{cls}_side_area", 0.8,
                      f"sum of NetSideArea of {len(els)} {CLASSES[cls][0]} {where}",
                      [Evidence("ifc", e.GlobalId, e.Name or cls, round(a, 3)) for e, a in ars], src)
    if qty == "length":
        ls = [(e, m.wall_length(e)) for e in els]
        return Answer(round(sum(v for _, v in ls), 2), "m", f"{cls}_length", 0.8,
                      f"sum of Length quantity of {len(els)} {CLASSES[cls][0]} {where}",
                      [Evidence("ifc", e.GlobalId, e.Name or cls, round(v, 3)) for e, v in ls], src)
    return Answer(len(els), "count", f"{cls}_count", 0.9, f"{len(els)} {CLASSES[cls][0]} {where}", _ev(m, els), src)
