"""Convert a layered floor-plan DXF (e.g. Stoa swiss_dwellings) into a 3D scene description.

The output is a JSON "scene" that the web viewer turns into 3D (and can export as .glb for Blender).
Every element keeps a stable ID and its source DXF layer, so a 3D object can always be traced back
to the exact geometry on the 2D drawing.

Rooms, doors, windows, apartments and fixtures come from ``stoa_qa.swiss.read_dxf`` — the same
read that answers questions. This file only adds what a 2D plan does not carry: assumed heights,
wall polygons, and the door lintel extruded across the wall.

    python blueprint3d/dxf_to_scene.py <plan.dxf> [more.dxf ...] [-o out_dir] [--storey 2.6]

Needs the project environment (ezdxf, shapely), same as ``stoa_qa``.
"""
import argparse, json, math, os, re, sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from shapely.geometry import Point

from stoa_qa.swiss import STAIRWELL, fixture_type, read_dxf

# Heights are NOT in a 2D plan. These are assumptions and are written into the scene so the UI shows them.
DEFAULTS = {"storey_height": 2.6, "door_head": 2.1, "window_sill": 0.9, "window_head": 2.1, "fixture_height": 0.9}

FIXTURE_LABEL = {"toilets": "Toilet", "basins": "Washbasin", "bathtubs": "Bathtub", "showers": "Shower"}
ROOM_KINDS = [  # (kind, label keywords) — DE / FR / IT, used for colouring
    ("bedroom", ["Zimmer", "Chambre", "Camera", "Stanza", "Pièce", "Schlafzimmer"]),
    ("living", ["Wohnen", "Essen", "Séjour", "Sejour", "Soggiorno", "Salon", "Pranzo"]),
    ("kitchen", ["Küche", "Kueche", "Cuisine", "Cucina"]),
    ("bath", ["Bad", "Dusche", "WC", "Salle de bain", "Bagno", "Douche", "Doccia"]),
    ("corridor", ["Korridor", "Gang", "Entrée", "Entree", "Couloir", "Corridoio", "Flur", "Vorplatz", "Disimpegno"]),
    ("storage", ["Reduit", "Réduit", "Abstell", "Ripostiglio", "Cave", "Keller"]),
    ("outdoor", ["Balkon", "Balcon", "Balcone", "Loggia", "Terrasse", "Terrazza", "Sitzplatz", "Wintergarten"]),
    ("stair", ["Treppe", "Escalier", "Scala", "Treppenhaus"]),
]


def inside(pt, poly):
    x, y = pt
    c = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1:
            c = not c
    return c


def r(v, n=4):
    return round(v, n)


def ring(poly):
    """Scene polygon: rounded, closing vertex dropped (the viewer closes it)."""
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    pts = list(poly.exterior.coords)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return [[r(x), r(y)] for x, y in pts]


def kind_of(label):
    if not label:
        return "other"
    for kind, words in ROOM_KINDS:
        if any(w.lower() in label.lower() for w in words):
            return kind
    return "other"


def wall_thickness(seg, u, n, wall_polys, step=0.01, max_t=0.6, default=0.2):
    """March from just beyond each end of the opening along n while still inside a wall polygon."""
    (ax, ay), (bx, by) = seg
    best = 0.0
    for (px, py), sgn in (((ax, ay), -1), ((bx, by), 1)):
        sx, sy = px + sgn * u[0] * 0.03, py + sgn * u[1] * 0.03
        t = step
        while t < max_t and any(inside((sx + n[0] * t, sy + n[1] * t), w) for w in wall_polys):
            t += step
        best = max(best, t - step)
    return best if best > 0.05 else default


def build_scene(path, heights):
    fl = read_dxf(str(Path(path).resolve()))
    walls = [{"id": f"wall:{h or i}", "layer": "WALL", "poly": ring(p)} for i, (h, p) in enumerate(fl.walls)]
    wall_polys = [w["poly"] for w in walls]

    seen = set()
    units = []
    for i, (h, p) in enumerate(fl.units):
        letter = next((l for l, pos in fl.apt_tags if p.contains(Point(pos))), None)
        label = f"Whg. {letter}" if letter else None
        apartment = bool(letter) and letter not in seen
        if letter:
            seen.add(letter)
        units.append({"id": f"unit:{h or i}", "layer": "UNIT_OUTLINE", "label": label, "apartment": apartment,
                      "poly": ring(p), "area_m2": r(p.area, 2)})

    room_out = []
    for rm in fl.rooms:
        if rm.number is None and rm.label is None:
            continue
        c = (rm.poly.centroid.x, rm.poly.centroid.y)
        anchor = rm.anchor or c
        unit = next((u["id"] for u, (_, p) in zip(units, fl.units) if p.contains(Point(anchor))), None)
        room_out.append({
            "id": f"room:{rm.number or rm.handle}", "layer": "ROOM_OUTLINE", "dxf_handle": rm.handle,
            "number": rm.number, "label": rm.label, "kind": kind_of(rm.label), "outdoor": rm.outdoor,
            "stairwell": (rm.label or "").strip().lower() in STAIRWELL,
            "area_m2": r(rm.area, 2), "printed_area": rm.printed_text, "unit": unit,
            "poly": ring(rm.poly), "holes": [ring(h) for h in rm.holes],
            "label_at": [r(anchor[0]), r(anchor[1])],
        })

    windows = [{"id": f"window:{h or i}", "layer": "WINDOW", "poly": ring(p)} for i, (h, p) in enumerate(fl.windows)]

    lines = [list(map(tuple, ln)) for ln in fl.door_lines if len(ln) == 2]
    doors = []
    for i, (h, leaf_poly) in enumerate(fl.doors):
        leaf = ring(leaf_poly)
        if len(leaf) < 4:
            continue
        lc = (leaf_poly.centroid.x, leaf_poly.centroid.y)
        best = min(lines, key=lambda ln: min(math.dist(q, v) for q in ln for v in leaf), default=None)
        door = {"id": f"door:{h or i}", "layer": "DOOR", "leaf": leaf}
        if best and min(math.dist(q, v) for q in best for v in leaf) < 0.15:
            (ax, ay), (bx, by) = best
            L = math.dist(best[0], best[1]) or 1
            ux, uy = (bx - ax) / L, (by - ay) / L
            nx, ny = -uy, ux
            mx, my = (ax + bx) / 2, (ay + by) / 2
            if (lc[0] - mx) * nx + (lc[1] - my) * ny > 0:  # point the normal away from the swing side
                nx, ny = -nx, -ny
            t = wall_thickness(best, (ux, uy), (nx, ny), wall_polys)
            door["opening"] = [[r(x), r(y)] for x, y in best]
            door["width_m"] = r(L, 3)
            door["lintel"] = [[r(x), r(y)] for x, y in (
                [ax, ay], [bx, by], [bx + nx * t, by + ny * t], [ax + nx * t, ay + ny * t])]
        doors.append(door)

    fixtures = []
    for i, f in enumerate(fl.fixtures):
        item = {"id": f"fixture:{f['handle'] or i}", "layer": "FIXTURE", "poly": ring(f["poly"])}
        label = FIXTURE_LABEL.get(fixture_type(f))
        if label:
            item["label"] = label
        fixtures.append(item)

    allpts = [q for g in (walls, windows) for el in g for q in el["poly"]] + [q for rm in room_out for q in rm["poly"]]
    if not allpts:
        allpts = [[0, 0], [1, 1]]
    xs, ys = [q[0] for q in allpts], [q[1] for q in allpts]
    rel = os.path.relpath(path).replace("\\", "/")
    m = re.search(r"data/.*", rel)
    return {
        "schema": "blueprint3d/scene@1",
        "source": {"file": m.group(0) if m else os.path.basename(path), "format": "dxf", "units": "m"},
        "assumptions": {**heights, "note": "Heights are assumed; a 2D plan does not contain them."},
        "bbox": [r(min(xs)), r(min(ys)), r(max(xs)), r(max(ys))],
        "counts": {"rooms": len(room_out), "walls": len(walls), "doors": len(doors),
                   "windows": len(windows), "units": len(fl.apt_tags), "fixtures": len(fixtures)},
        "units": units, "rooms": room_out, "walls": walls, "doors": doors, "windows": windows, "fixtures": fixtures,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dxf", nargs="+")
    ap.add_argument("-o", "--out", default=os.path.join(os.path.dirname(__file__), "viewer", "scenes"))
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    a = ap.parse_args()
    heights = {k: getattr(a, k) for k in DEFAULTS}
    os.makedirs(a.out, exist_ok=True)
    index_path = os.path.join(a.out, "index.json")
    index = json.load(open(index_path, encoding="utf-8")) if os.path.exists(index_path) else []
    for p in a.dxf:
        scene = build_scene(p, heights)
        name = os.path.splitext(os.path.basename(p))[0]
        with open(os.path.join(a.out, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(scene, f, ensure_ascii=False, separators=(",", ":"))
        index = [x for x in index if x["name"] != name] + [{"name": name, "source": scene["source"]["file"],
                                                            "counts": scene["counts"]}]
        c = scene["counts"]
        print(f"{name}: {c['rooms']} rooms, {c['walls']} walls, {c['doors']} doors, {c['windows']} windows, {c['units']} units")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(sorted(index, key=lambda x: x["name"]), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.exit(main())
