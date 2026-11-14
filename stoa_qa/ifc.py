"""ifc_models: answer from the IFC model (also for the PDF plans rendered from it).

Quantities come from IFC quantity sets when the authoring tool exported them,
otherwise from tessellated geometry. Evidence = IFC GlobalIds, which the 3D
viewer highlights directly.
"""
from __future__ import annotations

import math
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
    """The IFC model behind a dataset path: the file itself, or the model a storey plan was rendered from."""
    p = Path(path)
    if p.suffix.lower() == ".ifc":
        return DATASET / p
    stem = re.sub(r"_(plan|elevation|section)(-.*)?$", "", p.stem)
    for cand in (p.with_name(f"{stem}.ifc"), p.with_name(f"{stem}_model.ifc")):
        if (DATASET / cand).exists():
            return DATASET / cand
    return DATASET / p.with_name(f"{stem}.ifc")


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
        v = self.qto(el, "Length")
        if v:
            return v * self.length_scale
        g = self.shape(el)  # no quantity set: long side of the wall's footprint (metres)
        if g is None:
            return 0.0
        from shapely.geometry import MultiPoint

        xy = ushape.get_vertices(g)[:, :2]
        box = list(MultiPoint([tuple(p) for p in xy]).minimum_rotated_rectangle.exterior.coords)
        return max(math.dist(box[0], box[1]), math.dist(box[1], box[2])) if len(box) >= 3 else 0.0

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
    return [Evidence("ifc", e.GlobalId, label_fn(e), viewer_id=e.id()) for e in els]


def answer(path: str, question: str) -> Answer:
    ifc = ifc_path_for(path)
    if not ifc.exists():
        return Answer(None, None, "unknown", 0.0, f"IFC model {ifc.name} not found")
    m = load_model(str(ifc))
    src = ifc.relative_to(DATASET).as_posix()
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
                          f"room '{key}' {where}: {how}", [Evidence("ifc", sp.GlobalId, key, a, viewer_id=sp.id())], src)
        tot, ev = 0.0, []
        for sp in spaces:
            a, _ = m.space_area(sp)
            if a:
                tot += a
                ev.append(Evidence("ifc", sp.GlobalId, sp.LongName or sp.Name or "space", round(a, 2), viewer_id=sp.id()))
        return Answer(round(tot, 2), "m2", "total_space_area", 0.85, f"sum of {len(ev)} IfcSpace areas {where}", ev, src)

    cls = next((c for c in PRIORITY if c in objs), None)
    if cls is None:
        return Answer(None, None, "unknown", 0.0, "ifc: no element class recognised", source_file=src)
    els = m.elements(cls, storey)

    if qty == "volume":
        vols = [(e, m.volume(e)[0]) for e in els]
        return Answer(round(sum(v for _, v in vols), 2), "m3", f"{cls}_volume", 0.85,
                      f"sum of NetVolume (or mesh volume) of {len(els)} {CLASSES[cls][0]} {where}",
                      [Evidence("ifc", e.GlobalId, e.Name or cls, round(v, 3), viewer_id=e.id()) for e, v in vols], src)
    if qty == "area":
        ars = [(e, m.wall_side_area(e)) for e in els]
        return Answer(round(sum(a for _, a in ars), 2), "m2", f"{cls}_side_area", 0.8,
                      f"sum of NetSideArea of {len(els)} {CLASSES[cls][0]} {where}",
                      [Evidence("ifc", e.GlobalId, e.Name or cls, round(a, 3), viewer_id=e.id()) for e, a in ars], src)
    if qty == "length":
        ls = [(e, m.wall_length(e)) for e in els]
        return Answer(round(sum(v for _, v in ls), 2), "m", f"{cls}_length", 0.8,
                      f"sum of Length quantity of {len(els)} {CLASSES[cls][0]} {where}",
                      [Evidence("ifc", e.GlobalId, e.Name or cls, round(v, 3), viewer_id=e.id()) for e, v in ls], src)
    return Answer(len(els), "count", f"{cls}_count", 0.9, f"{len(els)} {CLASSES[cls][0]} {where}", _ev(m, els), src)


# ---------------------------------------------------------------- fact tables (for the query planner)

@lru_cache(maxsize=8)
def facts(ifc: str) -> dict:
    """Queryable tables for one model. Expensive columns (areas from geometry, quantity sets) are lazy."""
    m = load_model(ifc)
    sc = m.length_scale

    def storey(e):
        s = m.storey_of(e)
        return s.Name if s is not None else None

    def type_name(e):
        t = uel.get_type(e)
        return (t.Name if t is not None else None) or getattr(e, "ObjectType", None)

    def ev(e, label=None, value=None):
        return Evidence("ifc", e.GlobalId, label or e.Name or e.is_a(), value, viewer_id=e.id())

    def dim(e, attr):
        v = getattr(e, attr, None)
        return round(v * sc, 3) if isinstance(v, (int, float)) and v else None

    spaces = []
    for sp in m.elements("space"):
        key = m.space_keys(sp)[0] if m.space_keys(sp) else (sp.Name or "space")
        spaces.append({"name": key, "number": sp.Name, "storey": storey(sp), "_ev": ev(sp, key),
                       "_lazy": {"area_m2": (lambda sp=sp: round(m.space_area(sp)[0] or 0.0, 2))}})

    def openings(cls):
        return [{"name": e.Name, "storey": storey(e), "width_m": dim(e, "OverallWidth"), "height_m": dim(e, "OverallHeight"),
                 "type": type_name(e), "_ev": ev(e)} for e in m.elements(cls)]

    walls = [{"name": e.Name, "storey": storey(e), "type": type_name(e), "_ev": ev(e),
              "_lazy": {"length_m": (lambda e=e: round(m.wall_length(e), 3)),
                        "side_area_m2": (lambda e=e: round(m.wall_side_area(e), 3)),
                        "volume_m3": (lambda e=e: round(m.volume(e)[0], 3)),
                        "external": (lambda e=e: (uel.get_psets(e).get("Pset_WallCommon") or {}).get("IsExternal"))}}
             for e in m.elements("wall")]

    storeys = []
    for name, s in m.storeys.items():
        elev = s.Elevation * sc if isinstance(s.Elevation, (int, float)) else None
        storeys.append({"name": name, "elevation_m": None if elev is None else round(elev, 3),
                        "spaces": sum(1 for r in spaces if r["storey"] == name), "_ev": ev(s, name)})
    storeys.sort(key=lambda r: (r["elevation_m"] is None, r["elevation_m"] or 0))

    elements = []
    for e in m.f.by_type("IfcElement"):
        if e.Decomposes and e.Decomposes[0].RelatingObject.is_a(e.is_a()):
            continue
        elements.append({"ifc_class": e.is_a(), "name": e.Name, "storey": storey(e), "type": type_name(e), "_ev": ev(e)})
        if len(elements) >= 40000:
            break

    info = {"ifc_schema": m.f.schema, "file": Path(ifc).name}
    for app in m.f.by_type("IfcApplication")[:1]:
        info["authoring_application"] = f"{app.ApplicationFullName or ''} {app.Version or ''}".strip()
    for proj in m.f.by_type("IfcProject")[:1]:
        info["project_name"] = proj.LongName or proj.Name
    for b in m.f.by_type("IfcBuilding")[:1]:
        info["building_name"] = b.LongName or b.Name
    for s in m.f.by_type("IfcSite")[:1]:
        info["site_name"] = s.LongName or s.Name
    info["storeys"] = len(storeys)
    info["spaces"] = len(spaces)
    model = [{"key": k, "value": v} for k, v in info.items() if v not in (None, "")]

    def opening_text(r):
        size = " × ".join(f"{r[k]:.2f}" for k in ("width_m", "height_m") if r[k])
        return f"{r['name'] or 'opening'}" + (f" — {size} m" if size else "") + f" · {r['storey'] or 'no storey'}"

    return {
        "spaces": {"rows": spaces, "units": {"area_m2": "m2"}, "doc": "rooms (IfcSpace): name, number, storey, area_m2 (net floor area)",
                   "show": lambda r: f"{r['name']} — {r.get('area_m2', 0):.2f} m² · {r['storey'] or 'no storey'}"},
        "storeys": {"rows": storeys, "units": {"elevation_m": "m"}, "doc": "building storeys: name, elevation_m, spaces (count)",
                    "show": lambda r: f"{r['name']} — elevation {r['elevation_m']} m · {r['spaces']} rooms"},
        "doors": {"rows": openings("door"), "units": {"width_m": "m", "height_m": "m"}, "doc": "doors: name, storey, width_m, height_m, type",
                  "show": opening_text},
        "windows": {"rows": openings("window"), "units": {"width_m": "m", "height_m": "m"}, "doc": "windows: name, storey, width_m, height_m, type",
                    "show": opening_text},
        "walls": {"rows": walls, "units": {"length_m": "m", "side_area_m2": "m2", "volume_m3": "m3"},
                  "show": lambda r: f"{r['name'] or 'wall'} — {r.get('length_m', 0):.2f} m long · {r['storey'] or 'no storey'}",
                  "doc": "walls: name, storey, type, length_m, side_area_m2, volume_m3, external (true/false/null)"},
        "elements": {"rows": elements, "units": {}, "doc": "every building element: ifc_class (IfcWall, IfcSlab, IfcBeam, IfcColumn, IfcStair, "
                                                              "IfcRoof, IfcFurnishingElement, IfcFlowTerminal ...), name, storey, type"},
        "model": {"rows": model, "units": {}, "show": lambda r: f"{r['key'].replace('_', ' ')}: {r['value']}", "doc": "model metadata as key/value: ifc_schema, authoring_application, project_name, building_name, site_name, storeys, spaces"},
    }
