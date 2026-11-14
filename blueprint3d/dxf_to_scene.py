"""Convert a layered floor-plan DXF (e.g. Stoa swiss_dwellings) into a 3D scene description.

The output is a JSON "scene" that the web viewer turns into 3D (and can export as .glb for Blender).
Every element keeps a stable ID and its source DXF layer, so a 3D object can always be traced back
to the exact geometry on the 2D drawing.

    py -I dxf_to_scene.py <plan.dxf> [more.dxf ...] [-o out_dir] [--storey 2.6]

Standard library only. Coordinates are in metres when $INSUNITS = 6.
"""
import argparse, json, math, os, re, sys

INSUNITS = {0: None, 1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0}

# Heights are NOT in a 2D plan. These are assumptions and are written into the scene so the UI shows them.
DEFAULTS = {"storey_height": 2.6, "door_head": 2.1, "window_sill": 0.9, "window_head": 2.1, "fixture_height": 0.9}

MIN_APARTMENT_M2 = 15.0

OUTDOOR = {"Balkon", "Balcon", "Balcone", "Loggia", "Terrasse", "Terrazza", "Sitzplatz", "Jardin d'hiver",
           "Wintergarten", "Giardino d'inverno"}
ROOM_KINDS = [  # (kind, label keywords) — DE / FR / IT, used for colouring and question routing
    ("bedroom", ["Zimmer", "Chambre", "Camera", "Stanza", "Pièce", "Schlafzimmer"]),
    ("living", ["Wohnen", "Essen", "Séjour", "Sejour", "Soggiorno", "Salon", "Pranzo"]),
    ("kitchen", ["Küche", "Kueche", "Cuisine", "Cucina"]),
    ("bath", ["Bad", "Dusche", "WC", "Salle de bain", "Bagno", "Douche", "Doccia"]),
    ("corridor", ["Korridor", "Gang", "Entrée", "Entree", "Couloir", "Corridoio", "Flur", "Vorplatz", "Disimpegno"]),
    ("storage", ["Reduit", "Réduit", "Abstell", "Ripostiglio", "Cave", "Keller"]),
    ("outdoor", sorted(OUTDOOR)),
    ("stair", ["Treppe", "Escalier", "Scala", "Treppenhaus"]),
]


# ---------------------------------------------------------------- DXF reading

def read_dxf(path):
    raw = open(path, "rb").read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    L = [x.strip() for x in text.split("\n")]
    header = {}
    for key in ("$INSUNITS", "$ACADVER"):
        if key in L:
            header[key] = L[L.index(key) + 2]
    s = next(i for i in range(len(L) - 2) if L[i] == "SECTION" and L[i + 2] == "ENTITIES") + 3
    ents, cur = [], None
    for i in range(s, len(L) - 1, 2):
        code, val = L[i], L[i + 1]
        if code == "0":
            if val == "ENDSEC":
                break
            cur = {"type": val, "layer": "", "pts": [], "handle": None}
            ents.append(cur)
        elif cur is None:
            continue
        elif code == "5":
            cur["handle"] = val
        elif code == "8":
            cur["layer"] = val
        elif code == "1":
            cur["text"] = val
        elif code == "10":
            cur["pts"].append([float(val), None])
        elif code == "20" and cur["pts"] and cur["pts"][-1][1] is None:
            cur["pts"][-1][1] = float(val)
        elif code == "70":
            cur["flags"] = int(val)
        elif code == "40":
            cur.setdefault("r", float(val))
    scale = INSUNITS.get(int(header.get("$INSUNITS", 6)), 1.0) or 1.0
    for e in ents:
        e["pts"] = [[x * scale, y * scale] for x, y in e["pts"] if y is not None]
    return header, ents


# ---------------------------------------------------------------- 2D geometry helpers

def area(p):
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(p, p[1:] + p[:1]))) / 2


def inside(pt, poly):
    x, y = pt
    c = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c


def centroid(p):
    a = cx = cy = 0.0
    for (x1, y1), (x2, y2) in zip(p, p[1:] + p[:1]):
        f = x1 * y2 - x2 * y1
        a += f; cx += (x1 + x2) * f; cy += (y1 + y2) * f
    if abs(a) < 1e-12:
        return [sum(x for x, _ in p) / len(p), sum(y for _, y in p) / len(p)]
    return [cx / (3 * a), cy / (3 * a)]


def clean(p):
    """Drop a repeated closing vertex and near-duplicate points."""
    out = []
    for q in p:
        if not out or math.dist(q, out[-1]) > 1e-6:
            out.append(q)
    if len(out) > 2 and math.dist(out[0], out[-1]) < 1e-6:
        out.pop()
    return out


def r(v, n=4):
    return round(v, n)


def rp(p):
    return [[r(x), r(y)] for x, y in p]


def kind_of(label):
    if not label:
        return "other"
    for kind, words in ROOM_KINDS:
        if any(w.lower() in label.lower() for w in words):
            return kind
    return "other"


# ---------------------------------------------------------------- semantic extraction

def build_scene(path, heights):
    header, ents = read_dxf(path)
    poly = lambda layer: [e for e in ents if e["layer"] == layer and e["type"] == "LWPOLYLINE"]
    texts = [(e["text"], tuple(e["pts"][0]), e["handle"]) for e in ents
             if e["layer"] == "ROOM_TEXT" and e["type"] == "TEXT" and e["pts"]]

    walls = [{"id": f"wall:{e['handle'] or i}", "layer": "WALL", "poly": rp(clean(e["pts"]))}
             for i, e in enumerate(poly("WALL")) if len(clean(e["pts"])) >= 3]
    wall_polys = [w["poly"] for w in walls]

    # Rooms: ROOM_OUTLINE polygons; nested outlines are holes (columns, shafts)
    outlines = [(e, clean(e["pts"])) for e in poly("ROOM_OUTLINE")]
    outlines = [(e, p) for e, p in outlines if len(p) >= 3 and area(p) > 1e-4]
    rooms = []
    for i, (e, p) in enumerate(outlines):
        a = area(p)
        holes = [j for j, (_, o) in enumerate(outlines) if j != i and area(o) < a and all(inside(q, p) for q in o)]
        rooms.append({"handle": e["handle"] or str(i), "poly": p, "gross": a, "holes": holes, "texts": []})
    for s, pt, h in texts:
        cands = [rm for rm in rooms if inside(pt, rm["poly"])]
        if cands:
            min(cands, key=lambda rm: rm["gross"])["texts"].append((s, pt, h))
    units = []
    for i, e in enumerate(poly("UNIT_OUTLINE")):
        p = clean(e["pts"])
        if len(p) >= 3:
            label = next((s for s, pt, _ in texts if re.match(r"(Whg\.|Appt\.?|App\.)\s*[A-Z]", s) and inside(pt, p)), None)
            units.append({"id": f"unit:{e['handle'] or i}", "layer": "UNIT_OUTLINE", "label": label,
                          "poly": rp(p), "area_m2": r(area(p), 2)})
    # An apartment is a labelled outline (one per label, parts may be split) or an unlabelled one of real size;
    # tiny unlabelled outlines are drafting fragments.
    seen = set()
    for u in units:
        u["apartment"] = (u["label"] not in seen) if u["label"] else u["area_m2"] >= MIN_APARTMENT_M2
        if u["label"]:
            seen.add(u["label"])

    room_out = []
    for rm in rooms:
        direct = [j for j in rm["holes"] if not any(j in rooms[k]["holes"] for k in rm["holes"])]
        net = rm["gross"] - sum(rooms[j]["gross"] for j in direct)
        ts = [t for t, _, _ in rm["texts"]]
        number = next((t for t in ts if re.fullmatch(r"U?\d+\.\d+", t)), None)
        printed = next((t for t in ts if t.endswith(("m²", "m2"))), None)
        label = next((t for t in ts if t != number and t != printed and not re.match(r"(Whg\.|Appt\.?|App\.)\s*[A-Z]$", t)), None)
        if number is None and label is None:
            continue  # unlabelled outline: usually the outer envelope of a hole
        c = centroid(rm["poly"])
        unit = next((u["id"] for u in units if inside(c, u["poly"])), None)
        anchor = next((pt for t, pt, _ in rm["texts"] if t == number), None) or c
        room_out.append({
            "id": f"room:{number or rm['handle']}", "layer": "ROOM_OUTLINE", "dxf_handle": rm["handle"],
            "number": number, "label": label, "kind": kind_of(label), "outdoor": label in OUTDOOR,
            "area_m2": r(net, 2), "printed_area": printed, "unit": unit,
            "poly": rp(rm["poly"]), "holes": [rp(rooms[j]["poly"]) for j in direct],
            "label_at": [r(anchor[0]), r(anchor[1])],
        })

    windows = [{"id": f"window:{e['handle'] or i}", "layer": "WINDOW", "poly": rp(clean(e["pts"]))}
               for i, e in enumerate(poly("WINDOW")) if e.get("flags", 0) & 1 and len(clean(e["pts"])) >= 3]

    # Doors: a closed 5-point rectangle is the leaf; the 2-point line touching its hinge end is the opening.
    lines = [e["pts"] for e in poly("DOOR") if len(e["pts"]) == 2]
    doors = []
    for i, e in enumerate(poly("DOOR")):
        leaf = clean(e["pts"])
        if len(e["pts"]) != 5 or len(leaf) < 4:
            continue
        lc = centroid(leaf)
        best = min(lines, key=lambda ln: min(math.dist(q, v) for q in ln for v in leaf), default=None)
        door = {"id": f"door:{e['handle'] or i}", "layer": "DOOR", "leaf": rp(leaf)}
        if best and min(math.dist(q, v) for q in best for v in leaf) < 0.15:
            (ax, ay), (bx, by) = best
            L = math.dist(best[0], best[1]) or 1
            ux, uy = (bx - ax) / L, (by - ay) / L
            nx, ny = -uy, ux
            mx, my = (ax + bx) / 2, (ay + by) / 2
            if (lc[0] - mx) * nx + (lc[1] - my) * ny > 0:  # point the normal away from the swing side
                nx, ny = -nx, -ny
            t = wall_thickness(best, (ux, uy), (nx, ny), wall_polys)
            door["opening"] = rp(best)
            door["width_m"] = r(L, 3)
            door["lintel"] = rp([[ax, ay], [bx, by], [bx + nx * t, by + ny * t], [ax + nx * t, ay + ny * t]])
        doors.append(door)

    fixtures = [{"id": f"fixture:{e['handle'] or i}", "layer": "FIXTURE", "poly": rp(clean(e["pts"]))}
                for i, e in enumerate(poly("FIXTURE")) if e.get("flags", 0) & 1 and len(clean(e["pts"])) >= 3]

    allpts = [q for g in (walls, windows) for el in g for q in el["poly"]] + [q for rm in room_out for q in rm["poly"]]
    xs, ys = [q[0] for q in allpts], [q[1] for q in allpts]
    rel = os.path.relpath(path).replace("\\", "/")
    m = re.search(r"data/.*", rel)
    return {
        "schema": "blueprint3d/scene@1",
        "source": {"file": m.group(0) if m else os.path.basename(path), "format": "dxf",
                   "dxf_version": header.get("$ACADVER"), "units": "m"},
        "assumptions": {**heights, "note": "Heights are assumed; a 2D plan does not contain them."},
        "bbox": [r(min(xs)), r(min(ys)), r(max(xs)), r(max(ys))],
        "counts": {"rooms": len(room_out), "walls": len(walls), "doors": len(doors),
                   "windows": len(windows), "units": sum(u["apartment"] for u in units), "fixtures": len(fixtures)},
        "units": units, "rooms": room_out, "walls": walls, "doors": doors, "windows": windows, "fixtures": fixtures,
    }


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


# ---------------------------------------------------------------- CLI

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
