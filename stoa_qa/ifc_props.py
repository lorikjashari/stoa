"""IFC property sets as one queryable table for the planner: load-bearing, external, fire rating, materials, quantities...

The planner's IFC tables (stoa_qa.ifc.facts) cover rooms, storeys, openings, walls and element classes. Many
free-form questions ask for what the model stores in property sets instead: "which walls are load-bearing",
"fire rating of the doors", "materials of the exterior walls", "total slab volume". This module flattens those
into rows {ifc_class, name, type, storey, pset, prop, value}, one per element property (type properties are
inherited, as IfcOpenShell resolves them), plus one row per material layer. Noise property sets (graphics,
phasing, constraints, internal ids) are left out and the table is capped, so large models stay fast.

Values are the model's own: numbers in the model's units (the table's doc says so), text as written.
"""
from __future__ import annotations

import collections
import re
from functools import lru_cache

from . import ifc
from .core import Evidence

MAX_ROWS = 150_000
SKIP_PSET = re.compile(r"graphic|phasing|constraint|visibility|analytical|ifc parameters|^pset_revit_other$|text|overrides", re.I)
SKIP_PROP = {"id", "GlobalId", "Element ID", "Type Id", "Family and Type", "Category"}
# shown in the table's doc when present, so the planner knows these exist even among hundreds of property names
NOTABLE = ["FinishCeilingHeight", "ClearHeight", "Raumhöhe", "LoadBearing", "IsExternal", "FireRating", "FireExit", "AcousticRating", "ThermalTransmittance", "Material", "Material layer",
           "Structural Material", "Function", "Thickness", "Width", "Height", "Length", "NetVolume", "GrossVolume", "Volume", "NetArea",
           "NetSideArea", "GrossArea", "Area", "Cost", "Manufacturer", "Model", "Mark", "Description", "Glazing", "Handicap Accessible",
           "IsCombustible", "SurfaceSpreadOfFlame", "Status", "Reference", "Assembly Description", "Type Mark"]


def _material_rows(e, sc: float) -> list[tuple[str, str, object]]:
    import ifcopenshell.util.element as uel

    try:
        mat = uel.get_material(e, should_skip_usage=True)
    except Exception:
        return []
    if mat is None:
        return []
    out: list[tuple[str, str, object]] = []
    if mat.is_a("IfcMaterial"):
        out.append(("Material", "Material", mat.Name))
    elif mat.is_a("IfcMaterialLayerSet"):
        for i, lay in enumerate(mat.MaterialLayers or []):
            name = lay.Material.Name if lay.Material else "?"
            thick = f" {lay.LayerThickness * sc:.3f} m" if isinstance(lay.LayerThickness, (int, float)) else ""
            out += [("Material", "Material", name), ("Material", "Material layer", f"{i + 1}. {name}{thick}")]
    elif mat.is_a("IfcMaterialList"):
        out += [("Material", "Material", m.Name) for m in mat.Materials or []]
    elif mat.is_a("IfcMaterialConstituentSet"):
        out += [("Material", "Material", c.Material.Name) for c in mat.MaterialConstituents or [] if c.Material]
    elif mat.is_a("IfcMaterialProfileSet"):
        out += [("Material", "Material", p.Material.Name) for p in mat.MaterialProfiles or [] if p.Material]
    return out


@lru_cache(maxsize=8)
def table(path: str) -> dict:
    """The `properties` table for one IFC model (cached per file)."""
    import ifcopenshell.util.element as uel

    m = ifc.load_model(path)
    sc = m.length_scale
    scale = _unit_scales(m.f, sc)
    rows: list[dict] = []
    seen_props: collections.Counter = collections.Counter()
    # rooms first (ceiling and clear heights, finishes, occupancy live on IfcSpace), then the building elements
    for e in list(m.f.by_type("IfcSpace")) + list(m.f.by_type("IfcElement")):
        if e.is_a("IfcElement") and e.Decomposes and e.Decomposes[0].RelatingObject.is_a(e.is_a()):
            continue  # parts of an element of the same class (stair flights, curtain wall panels) are counted with the parent
        s = m.storey_of(e)
        t = uel.get_type(e)
        room = e.is_a("IfcSpace")
        label = " ".join(x for x in (e.Name, getattr(e, "LongName", None)) if x) if room else e.Name
        base = {"ifc_class": e.is_a(), "name": label, "type": ("room " + (label or "")) if room else
                ((t.Name if t is not None else None) or getattr(e, "ObjectType", None)),
                "storey": s.Name if s is not None else None}
        ev = Evidence("ifc", e.GlobalId, e.Name or e.is_a(), None, viewer_id=e.id())
        try:
            psets = uel.get_psets(e)
        except Exception:
            psets = {}
        props = [(ps, k, v) for ps, d in psets.items() if ps and not SKIP_PSET.search(ps)
                 for k, v in d.items() if k not in SKIP_PROP and v not in (None, "")]
        props += _material_rows(e, sc)
        for ps, k, v in props:
            if placeholder(k, v):
                continue
            unit = None
            if isinstance(v, (int, float)) and not isinstance(v, bool) and (unit := unit_of(k)):
                v = v * scale[unit]  # the model's own length / area / volume unit -> m, m2, m3
            if isinstance(v, float):
                v = round(v, 4)
            elif not isinstance(v, (int, bool, str)):
                v = str(v)
            rows.append({**base, "pset": ps, "prop": k, "value": v, "unit": unit, "_ev": ev})
            seen_props[k] += 1
        if len(rows) >= MAX_ROWS:
            break
    notable = [f"{k} ({seen_props[k]})" for k in NOTABLE if seen_props.get(k)]
    doc = ("element properties from the model's property sets and materials, one row per element property: "
           "ifc_class, name, type, storey, pset, prop, value. Filter on prop (exact property name) and ifc_class; "
           "lengths are in m, areas in m2, volumes in m3 (unit column); other numbers as stored. Properties present include: " + ", ".join(notable or [k for k, _ in seen_props.most_common(20)]))
    return {"rows": rows, "units": {}, "doc": doc,
            # one line per type, not per element: identical lines are collapsed with a count by the planner
            "show": lambda r: f"{r['type'] or r['name'] or r['ifc_class']} — {r['prop']}: {r['value']}" + (f" {UNIT_TEXT[r['unit']]}" if r.get('unit') else "")}


LENGTH = re.compile(r"height|höhe|hoehe|length|länge|laenge|width|breite|thickness|dicke|stärke|depth|tiefe|perimeter|umfang|elevation|offset|radius|diameter|durchmesser", re.I)
AREA = re.compile(r"area|fläche|flaeche", re.I)
VOLUME = re.compile(r"volume|volumen|rauminhalt", re.I)
UNIT_TEXT = {"m": "m", "m2": "m²", "m3": "m³"}


def unit_of(prop: str) -> str | None:
    """m / m2 / m3 for a property whose name says it is a length, area or volume (checked in that order: volume, area, length)."""
    if VOLUME.search(prop):
        return "m3"
    if AREA.search(prop):
        return "m2"
    if LENGTH.search(prop):
        return "m"
    return None


def _unit_scales(f, length_scale: float) -> dict[str, float]:
    """Model unit -> SI factor per kind. Area and volume use the model's own declared units (a model in millimetres
    often still states areas in m2), falling back to the length scale squared / cubed."""
    out = {"m": length_scale, "m2": length_scale ** 2, "m3": length_scale ** 3}
    try:
        import ifcopenshell.util.unit as uu

        out["m2"] = uu.calculate_unit_scale(f, "AREAUNIT")
        out["m3"] = uu.calculate_unit_scale(f, "VOLUMEUNIT")
    except Exception:
        pass
    return out


EMPTY = {"none", "keine", "kein", "nicht definiert", "undefined", "not defined", "n/a", "na", "-", "--", "default", "<none>", "unknown", "0 - none"}


def placeholder(prop: str, value) -> bool:
    """An exported property with no real content: its value repeats the property's own name ('FireRating: Fire Rating',
    'ExpectedLife: ExpectedLife', 'Material: Material') or says none ('Keine', 'Nicht definiert')."""
    if not isinstance(value, str):
        return False
    v = "".join(value.lower().split())
    return not v or v.strip("<>()") in {e.replace(" ", "") for e in EMPTY} or v == "".join(prop.lower().split())


# ---- class inheritance for ifc_class filters: asking for IfcFlowTerminal also finds IfcSanitaryTerminal (IFC4 subtypes)

@lru_cache(maxsize=4096)
def ancestors(cls: str) -> frozenset[str]:
    """Lower-case names of the class and all its supertypes, from IfcOpenShell's schemas."""
    import ifcopenshell

    for schema in ("IFC4", "IFC2X3", "IFC4X3"):
        try:
            d = ifcopenshell.ifcopenshell_wrapper.schema_by_name(schema).declaration_by_name(cls)
        except Exception:
            continue
        out = set()
        while d is not None:
            out.add(d.name().lower())
            d = d.supertype()
        return frozenset(out)
    return frozenset({cls.lower()})


def class_match(v, op: str, val) -> bool | None:
    """ifc_class filter with inheritance; None when this is not an ifc_class comparison it can decide."""
    if not isinstance(v, str):
        return None
    want = val if isinstance(val, list) else [val]
    want = [str(w).strip().lower() for w in want]
    want = [w if w.startswith("ifc") else f"ifc{w}" for w in want]
    anc = ancestors(v)
    if op in ("=", "in"):
        return any(w in anc for w in want)
    if op == "!=":
        return not any(w in anc for w in want)
    if op == "contains":
        return any(w.removeprefix("ifc") in a for w in want for a in anc)
    return None
