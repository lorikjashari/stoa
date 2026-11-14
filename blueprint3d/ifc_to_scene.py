"""Read an IFC building model into a semantic scene: storeys, rooms, elements and their quantities.

The 3D geometry itself is rendered in the browser by web-ifc; this file supplies the facts. Both sides use the
IFC entity number (#123 = "expressID"), so every answer can be highlighted on the exact 3D element it came from.

Rooms, storeys and quantities come from ``stoa_qa.ifc`` — the same IfcOpenShell model that answers questions.
This file only adds what the viewer needs on top: per-storey plan filenames and the storey of every other product.

    python blueprint3d/ifc_to_scene.py <model.ifc> [more.ifc ...] [-o out_dir]

Needs the project environment (ifcopenshell), same as ``stoa_qa``.
"""
import argparse, json, os, re, sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stoa_qa.ifc import load_model

# answer class -> scene group. Extras below are drawn, never asked about.
GROUPS = [
    ("wall", "walls"), ("door", "doors"), ("window", "windows"), ("slab", "slabs"),
    ("column", "columns"), ("beam", "beams"), ("stair", "stairs"), ("sanitary", "sanitary"), ("light", "lights"),
]
EXTRA = [("roofs", ["IfcRoof"]), ("railings", ["IfcRailing"]), ("furniture", ["IfcFurnishingElement"])]
QTO_NAMES = {
    "gross_floor_area_m2": ("GrossFloorArea", "Brutto-Grundfläche", "Area", "Fläche"),
    "height_m": ("Height", "FinishCeilingHeight", "Höhe"),
    "slab_area": ("NetArea",),
}


def rnd(v, n=4):
    return None if v is None else round(float(v), n)


def build_scene(path):
    m = load_model(str(path))
    E = m.f
    storey_ids = {s.id(): s for s in E.by_type("IfcBuildingStorey")}

    storeys = []
    for s in storey_ids.values():
        elev = s.Elevation if isinstance(s.Elevation, (int, float)) else None
        storeys.append({"id": s.id(), "name": s.Name, "long_name": getattr(s, "LongName", None),
                        "elevation_m": rnd(elev * m.length_scale, 3) if elev is not None else None})
    storeys.sort(key=lambda s: (s["elevation_m"] is None, s["elevation_m"] or 0))
    base, folder = os.path.splitext(os.path.basename(path))[0], os.path.dirname(path)
    plans = sorted(f for f in os.listdir(folder) if f.startswith(base + "_plan-") and f.endswith(".svg")) if os.path.isdir(folder) else []
    slug = lambda s: re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")
    for s in storeys:
        hit = [f for f in plans if f[len(base) + 6:-4].split("-", 1)[-1] == slug(s["name"])]
        s["plan"] = hit[0] if hit else None

    def storey_id(el):
        s = m.storey_of(el)
        return s.id() if s is not None else None

    rooms = []
    for sp in m.elements("space"):
        name, long_name = sp.Name, sp.LongName
        label = f"{long_name} ({name})" if long_name and name else (long_name or name)
        area, how = m.space_area(sp)
        rooms.append({
            "id": sp.id(), "name": name, "long_name": long_name, "label": label, "storey": storey_id(sp),
            "net_floor_area_m2": rnd(area),
            "area_source": "quantity set" if how.startswith("IFC quantity") else "geometry footprint" if area else None,
            "gross_floor_area_m2": rnd(m.qto(sp, *QTO_NAMES["gross_floor_area_m2"])),
            "height_m": rnd(m.qto(sp, *QTO_NAMES["height_m"])),
        })

    elements = []
    for cls, group in GROUPS:
        for e in m.elements(cls):
            el = {"id": e.id(), "ifc_type": e.is_a(), "group": group, "name": e.Name, "storey": storey_id(e)}
            if group == "walls":
                el.update(length_m=rnd(m.wall_length(e)), net_side_area_m2=rnd(m.wall_side_area(e)), net_volume_m3=rnd(m.volume(e)[0]))
            elif group == "slabs":
                el.update(net_volume_m3=rnd(m.volume(e)[0]), net_area_m2=rnd(m.qto(e, *QTO_NAMES["slab_area"])))
            elements.append(el)
    for group, types in EXTRA:
        for e in m.products(types):
            elements.append({"id": e.id(), "ifc_type": e.is_a(), "group": group, "name": e.Name, "storey": storey_id(e)})

    known = {r["id"] for r in rooms} | {e["id"] for e in elements} | set(storey_ids)
    other = {}
    for e in E.by_type("IfcProduct"):
        i = e.id()
        if i in known:
            continue
        s = storey_id(e)
        if s is not None:
            other.setdefault(str(s), []).append(i)

    proj = next(iter(E.by_type("IfcProject")), None)
    bldg = next(iter(E.by_type("IfcBuilding")), None)
    rel = os.path.relpath(path).replace("\\", "/")
    src = re.search(r"data/.*", rel)
    counts = {"rooms": len(rooms), "storeys": len(storeys)}
    for el in elements:
        counts[el["group"]] = counts.get(el["group"], 0) + 1
    scale = {"length": m.length_scale}
    try:
        import ifcopenshell.util.unit as uunit
        scale["area"] = uunit.calculate_unit_scale(E, "AREAUNIT")
        scale["volume"] = uunit.calculate_unit_scale(E, "VOLUMEUNIT")
    except Exception:
        scale.setdefault("area", 1.0)
        scale.setdefault("volume", 1.0)
    return {
        "schema": "blueprint3d/scene@1",
        "source": {"file": src.group(0) if src else os.path.basename(path), "format": "ifc",
                   "ifc_schema": E.schema, "units": "m"},
        "names": {"project": proj.Name if proj else None, "project_long": getattr(proj, "LongName", None) if proj else None,
                  "building": bldg.Name if bldg else None, "building_long": getattr(bldg, "LongName", None) if bldg else None},
        "unit_scale": scale, "counts": counts, "storeys": storeys, "rooms": rooms, "elements": elements,
        "other_by_storey": other,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ifc", nargs="+")
    ap.add_argument("-o", "--out", default=os.path.join(os.path.dirname(__file__), "viewer", "scenes"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    index_path = os.path.join(a.out, "index.json")
    index = json.load(open(index_path, encoding="utf-8")) if os.path.exists(index_path) else []
    for p in a.ifc:
        scene = build_scene(p)
        name = os.path.splitext(os.path.basename(p))[0]
        with open(os.path.join(a.out, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(scene, f, ensure_ascii=False, separators=(",", ":"))
        index = [x for x in index if x["name"] != name] + [{"name": name, "source": scene["source"]["file"], "counts": scene["counts"]}]
        print(f"{name}: {scene['counts']}")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(sorted(index, key=lambda x: x["name"]), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.exit(main())
