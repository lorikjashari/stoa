"""Read an IFC (STEP) building model into a semantic scene: storeys, rooms, elements and their quantities.

The 3D geometry itself is rendered in the browser by web-ifc; this file supplies the facts. Both sides use the
IFC entity number (#123 = "expressID"), so every answer can be highlighted on the exact 3D element it came from.

    py -I ifc_to_scene.py <model.ifc> [more.ifc ...] [-o out_dir]

Standard library only; streams the file and parses just the entity types it needs.
"""
import argparse, json, os, re, sys

ELEMENT_TYPES = {  # IFC class -> group name used by the viewer and the question router
    "IFCWALL": "walls", "IFCWALLSTANDARDCASE": "walls", "IFCWALLELEMENTEDCASE": "walls",
    "IFCDOOR": "doors", "IFCDOORSTANDARDCASE": "doors",
    "IFCWINDOW": "windows", "IFCWINDOWSTANDARDCASE": "windows",
    "IFCSLAB": "slabs", "IFCSLABSTANDARDCASE": "slabs", "IFCSLABELEMENTEDCASE": "slabs",
    "IFCCOLUMN": "columns", "IFCCOLUMNSTANDARDCASE": "columns",
    "IFCBEAM": "beams", "IFCBEAMSTANDARDCASE": "beams",
    "IFCSTAIR": "stairs", "IFCROOF": "roofs", "IFCRAILING": "railings",
    "IFCSANITARYTERMINAL": "sanitary", "IFCFURNISHINGELEMENT": "furniture", "IFCFURNITURE": "furniture",
}
QUANTITY_TYPES = {"IFCQUANTITYLENGTH": "length", "IFCQUANTITYAREA": "area", "IFCQUANTITYVOLUME": "volume",
                  "IFCQUANTITYCOUNT": "count"}
NEEDED = set(ELEMENT_TYPES) | set(QUANTITY_TYPES) | {
    "IFCPROJECT", "IFCBUILDING", "IFCBUILDINGSTOREY", "IFCSPACE", "IFCRELAGGREGATES",
    "IFCRELCONTAINEDINSPATIALSTRUCTURE", "IFCRELDEFINESBYPROPERTIES", "IFCELEMENTQUANTITY", "IFCUNITASSIGNMENT",
    "IFCSIUNIT", "IFCCONVERSIONBASEDUNIT", "IFCMEASUREWITHUNIT", "IFCRELNESTS"}
SI_PREFIX = {"MILLI": 1e-3, "CENTI": 1e-2, "DECI": 1e-1, "KILO": 1e3, None: 1.0}


# ---------------------------------------------------------------- STEP parsing

def decode(s):
    """IFC string escapes: \\X2\\00F6\\X0\\, \\X\\E4, \\S\\x, and doubled quotes."""
    s = re.sub(r"\\X2\\((?:[0-9A-F]{4})+)\\X0\\", lambda m: "".join(chr(int(m[1][i:i + 4], 16)) for i in range(0, len(m[1]), 4)), s)
    s = re.sub(r"\\X4\\((?:[0-9A-F]{8})+)\\X0\\", lambda m: "".join(chr(int(m[1][i:i + 8], 16)) for i in range(0, len(m[1]), 8)), s)
    s = re.sub(r"\\X\\([0-9A-F]{2})", lambda m: bytes([int(m[1], 16)]).decode("latin-1"), s)
    s = re.sub(r"\\S\\(.)", lambda m: chr(ord(m[1]) + 128), s)
    return s.replace("''", "'")


TOKEN = re.compile(r"\s*(?:(?P<str>'(?:[^']|'')*')|(?P<ref>#\d+)|(?P<enum>\.[A-Z0-9_]+\.)|(?P<num>[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
                   r"|(?P<typed>[A-Z0-9_]+)\(|(?P<open>\()|(?P<close>\))|(?P<null>[$*])|(?P<comma>,))")


def parse_args(body):
    stack, cur, pos = [], [], 0
    while pos < len(body):
        m = TOKEN.match(body, pos)
        if not m:
            pos += 1
            continue
        pos = m.end()
        k = m.lastgroup
        if k == "str": cur.append(decode(m[k][1:-1]))
        elif k == "ref": cur.append(int(m[k][1:]))
        elif k == "enum": cur.append(m[k])
        elif k == "num": cur.append(float(m[k]))
        elif k == "null": cur.append(None)
        elif k in ("open", "typed"):
            stack.append(cur); cur = []
            if k == "typed": cur.append(("TYPED", m["typed"]))
        elif k == "close":
            if not stack: break
            inner = cur; cur = stack.pop()
            if inner and isinstance(inner[0], tuple) and inner[0][0] == "TYPED":
                cur.append(inner[1] if len(inner) > 1 else None)  # IFCLENGTHMEASURE(0.3) -> 0.3
            else:
                cur.append(inner)
    return cur


class Model(dict):
    """Lazy STEP store: indexes every entity's type and offset, parses an entity only when it is read."""

    def __init__(self, path):
        super().__init__()
        self.text = open(path, "rb").read().decode("latin-1")
        self.schema = (re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'", self.text) or [None, None])[1]
        self.index = {}
        for m in re.finditer(r"#(\d+)\s*=\s*([A-Z0-9_]+)\s*\(", self.text):
            self.index[int(m[1])] = (m[2], m.end())

    def __missing__(self, i):
        if i not in self.index:
            raise KeyError(i)
        t, start = self.index[i]
        end = self.text.find(");", start)
        body = self.text[start:end]
        while body.count("'") % 2 and end != -1:  # a ");" inside a string would cut early
            end = self.text.find(");", end + 2)
            body = self.text[start:end]
        v = (t, parse_args(body))
        self[i] = v
        return v

    def get(self, i, default=None):
        try:
            return self[i]
        except KeyError:
            return default

    def of_type(self, *types):
        return [i for i, (t, _) in self.index.items() if t in types]


def read_ifc(path):
    m = Model(path)
    return m.schema, m


# ---------------------------------------------------------------- footprint area from geometry (no kernel)

def shoelace(pts):
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))) / 2


def curve_points(E, c, depth=0):
    """2D points of a profile curve: polyline, indexed polycurve, composite curve (chords for arcs)."""
    if depth > 10 or c is None:
        return []
    t, a = E.get(c, (None, None))
    if t == "IFCPOLYLINE":
        return [E[p][1][0][:2] for p in a[0]]
    if t == "IFCINDEXEDPOLYCURVE":
        pts = E[a[0]][1][0]
        return [p[:2] for p in pts]
    if t == "IFCCOMPOSITECURVE":
        out = []
        for seg in a[0]:
            st, sa = E.get(seg, (None, None))
            if st in ("IFCCOMPOSITECURVESEGMENT", "IFCREPARAMETRISEDCOMPOSITECURVESEGMENT"):
                pts = curve_points(E, sa[2], depth + 1)
                if sa[1] in (".F.", False):
                    pts = pts[::-1]
                out += pts
        return out
    if t == "IFCTRIMMEDCURVE":  # chord between the trimming points when they are cartesian
        pts = [E[x][1][0][:2] for trim in (a[1], a[2]) for x in trim if isinstance(x, int) and E.get(x, ("",))[0] == "IFCCARTESIANPOINT"]
        return pts if a[3] in (".T.", True) else pts[::-1]
    return []


def profile_area(E, p, depth=0):
    t, a = E.get(p, (None, None))
    if t == "IFCRECTANGLEPROFILEDEF":
        return a[3] * a[4]
    if t == "IFCCIRCLEPROFILEDEF":
        return 3.141592653589793 * a[3] ** 2
    if t == "IFCARBITRARYCLOSEDPROFILEDEF":
        return shoelace(curve_points(E, a[2]))
    if t == "IFCARBITRARYPROFILEDEFWITHVOIDS":
        return shoelace(curve_points(E, a[2])) - sum(shoelace(curve_points(E, v)) for v in a[3] or [])
    if t == "IFCDERIVEDPROFILEDEF":
        return profile_area(E, a[2], depth + 1)
    return None


def brep_footprint(E, shell):
    """Sum of plan-projected areas of downward-facing faces of a faceted shell."""
    total = 0.0
    for f in E[shell][1][0]:
        loops = []  # (is_outer, newell normal) per bound
        for b in E[f][1][0]:
            bt, ba = E[b]
            lt, la = E.get(ba[0], (None, None))
            if lt != "IFCPOLYLOOP":
                continue
            P = [E[q][1][0] for q in la[0]]
            if len(P) < 3 or len(P[0]) < 3:
                continue
            n = [0.0, 0.0, 0.0]  # Newell's method; |n| = 2 x polygon area
            for (x1, y1, z1), (x2, y2, z2) in zip(P, P[1:] + P[:1]):
                n[0] += (y1 - y2) * (z1 + z2); n[1] += (z1 - z2) * (x1 + x2); n[2] += (x1 - x2) * (y1 + y2)
            loops.append((bt == "IFCFACEOUTERBOUND", n))
        if not loops:
            continue
        outer = next((n for o, n in loops if o), loops[0][1])
        mag = sum(c * c for c in outer) ** .5 or 1
        if outer[2] / mag < -0.7:  # face points down: it is part of the floor footprint
            total += abs(outer[2]) / 2 - sum(abs(n[2]) / 2 for o, n in loops if n is not outer)
    return total


def item_footprint(E, item, depth=0):
    if depth > 8:
        return None
    t, a = E.get(item, (None, None))
    if t == "IFCEXTRUDEDAREASOLID":
        d = E.get(a[2], (None, [[0, 0, 1]]))[1][0]
        return profile_area(E, a[0]) if abs(d[2]) > 0.7 else None
    if t in ("IFCFACETEDBREP", "IFCFACETEDBREPWITHVOIDS"):
        return brep_footprint(E, a[0])
    if t in ("IFCBOOLEANCLIPPINGRESULT", "IFCBOOLEANRESULT"):
        return item_footprint(E, a[1], depth + 1)
    if t == "IFCMAPPEDITEM":
        rep = E[E[a[0]][1][1]][1]
        vals = [item_footprint(E, x, depth + 1) for x in rep[3]]
        return sum(v for v in vals if v) or None
    if t == "IFCSHELLBASEDSURFACEMODEL":
        return sum(brep_footprint(E, s) for s in a[0]) or None
    return None


def space_footprint_area(E, space_id):
    """Plan area of an IfcSpace's Body geometry, in project units squared."""
    rep = E[space_id][1][6]
    if rep is None:
        return None
    for r in E[rep][1][2]:
        rt, ra = E.get(r, (None, None))
        if rt == "IFCSHAPEREPRESENTATION" and ra[1] == "Body":
            vals = [item_footprint(E, it) for it in ra[3]]
            vals = [v for v in vals if v]
            if vals:
                return sum(vals)
    return None


# ---------------------------------------------------------------- model -> scene

def unit_scales(ents):
    scale = {"length": 1.0, "area": 1.0, "volume": 1.0}
    kind = {".LENGTHUNIT.": "length", ".AREAUNIT.": "area", ".VOLUMEUNIT.": "volume"}
    assign = next((ents[i][1] for i in ents.of_type("IFCUNITASSIGNMENT")), None)
    for ref in (assign[0] if assign else []):
        t, a = ents.get(ref, (None, None))
        if t == "IFCSIUNIT" and a[1] in kind:
            prefix = a[2][1:-1] if a[2] else None
            p = SI_PREFIX.get(prefix, 1.0)
            scale[kind[a[1]]] = p ** {"length": 1, "area": 2, "volume": 3}[kind[a[1]]]
        elif t == "IFCCONVERSIONBASEDUNIT" and a[1] in kind:
            mt, ma = ents.get(a[3], (None, None))
            if mt == "IFCMEASUREWITHUNIT" and isinstance(ma[0], float):
                factor = ma[0]
                st, sa = ents.get(ma[1], (None, None))
                if st == "IFCSIUNIT" and sa[2]:
                    factor *= SI_PREFIX.get(sa[2][1:-1], 1.0) ** {"length": 1, "area": 2, "volume": 3}[kind[a[1]]]
                scale[kind[a[1]]] = factor
    return scale


def build_scene(path):
    schema, E = read_ifc(path)
    scale = unit_scales(E)
    by_type = {}
    for i, (t, _) in E.index.items():
        if t in NEEDED:
            by_type.setdefault(t, []).append(i)

    # quantities per object: {id: {qset: {name: value_si}}}
    qty = {}
    for i in by_type.get("IFCRELDEFINESBYPROPERTIES", []):
        a = E[i][1]
        objs, pdef = a[4], a[5]
        if E.get(pdef, ("",))[0] != "IFCELEMENTQUANTITY":
            continue
        qa = E[pdef][1]
        values = {}
        for q in qa[5] or []:
            qt, qargs = E.get(q, (None, None))
            if qt in QUANTITY_TYPES and len(qargs) > 3 and isinstance(qargs[3], float):
                k = QUANTITY_TYPES[qt]
                values[qargs[0]] = qargs[3] * scale.get(k, 1.0)
        for o in objs or []:
            qty.setdefault(o, {}).setdefault(qa[2] or "", {}).update(values)

    # spatial tree
    parent = {}
    for i in by_type.get("IFCRELAGGREGATES", []) + by_type.get("IFCRELNESTS", []):
        a = E[i][1]
        for c in a[5] or []:
            parent.setdefault(c, a[4])
    for i in by_type.get("IFCRELCONTAINEDINSPATIALSTRUCTURE", []):
        a = E[i][1]
        for c in a[4] or []:
            parent[c] = a[5]
    storey_ids = by_type.get("IFCBUILDINGSTOREY", [])

    def storey_of(i, depth=0):
        while i is not None and depth < 20:
            if i in storey_ids:
                return i
            i, depth = parent.get(i), depth + 1
        return None

    storeys = []
    for i in storey_ids:
        a = E[i][1]
        elev = a[9] if len(a) > 9 and isinstance(a[9], float) else None
        storeys.append({"id": i, "name": a[2], "long_name": a[7], "elevation_m": round(elev * scale["length"], 3) if elev is not None else None})
    storeys.sort(key=lambda s: (s["elevation_m"] is None, s["elevation_m"] or 0))
    # the dataset ships plans rendered per storey: <model>_plan-NN-<storey-slug>.svg
    base, folder = os.path.splitext(os.path.basename(path))[0], os.path.dirname(path)
    plans = sorted(f for f in os.listdir(folder) if f.startswith(base + "_plan-") and f.endswith(".svg"))
    slug = lambda s: re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")
    for s in storeys:
        hit = [f for f in plans if f[len(base) + 6:-4].split("-", 1)[-1] == slug(s["name"])]
        s["plan"] = hit[0] if hit else None

    def q(i, *names):
        # standard quantity sets first (Qto_* / BaseQuantities), vendor sets (ArchiCADQuantities, ...) as fallback
        sets = sorted(qty.get(i, {}).items(), key=lambda kv: not (kv[0].startswith("Qto_") or kv[0] == "BaseQuantities"))
        for _, qs in sets:
            for n in names:
                if n in qs:
                    return qs[n]
        return None

    rooms = []
    for i in by_type.get("IFCSPACE", []):
        a = E[i][1]
        name, long_name = a[2], a[7]
        label = f"{long_name} ({name})" if long_name and name else (long_name or name)
        stored = q(i, "NetFloorArea", "Netto-Grundfläche", "NetArea")
        geom = None
        if stored is None:
            try:
                g = space_footprint_area(E, i)
                geom = g * scale["length"] ** 2 if g else None
            except (KeyError, IndexError, TypeError):
                geom = None
        rooms.append({"id": i, "name": name, "long_name": long_name, "label": label, "storey": storey_of(i),
                      "net_floor_area_m2": rnd(stored if stored is not None else geom),
                      "area_source": "quantity set" if stored is not None else "geometry footprint" if geom else None,
                      "gross_floor_area_m2": rnd(q(i, "GrossFloorArea", "Brutto-Grundfläche", "Area", "Fläche")),
                      "height_m": rnd(q(i, "Height", "FinishCeilingHeight", "Höhe"))})

    elements = []
    for t, group in ELEMENT_TYPES.items():
        for i in by_type.get(t, []):
            a = E[i][1]
            el = {"id": i, "ifc_type": t, "group": group, "name": a[2] if len(a) > 2 else None, "storey": storey_of(i)}
            if group == "walls":
                el.update(length_m=rnd(q(i, "Length")), net_side_area_m2=rnd(q(i, "NetSideArea")), net_volume_m3=rnd(q(i, "NetVolume")))
            elif group == "slabs":
                el.update(net_volume_m3=rnd(q(i, "NetVolume")), net_area_m2=rnd(q(i, "NetArea")))
            elements.append(el)

    # every other placed product (members, plates, coverings, openings...) -> storey, so the viewer can isolate floors
    known = {r["id"] for r in rooms} | {e["id"] for e in elements} | set(storey_ids)
    other = {}
    for c in parent:
        if c not in known:
            s = storey_of(c)
            if s is not None:
                other.setdefault(str(s), []).append(c)

    proj = next((E[i][1] for i in by_type.get("IFCPROJECT", [])), None)
    bldg = next((E[i][1] for i in by_type.get("IFCBUILDING", [])), None)
    rel = os.path.relpath(path).replace("\\", "/")
    m = re.search(r"data/.*", rel)
    counts = {"rooms": len(rooms), "storeys": len(storeys)}
    for el in elements:
        counts[el["group"]] = counts.get(el["group"], 0) + 1
    return {
        "schema": "blueprint3d/scene@1",
        "source": {"file": m.group(0) if m else os.path.basename(path), "format": "ifc", "ifc_schema": schema, "units": "m"},
        "names": {"project": proj[2] if proj else None, "project_long": proj[3] if proj else None,
                  "building": bldg[2] if bldg else None, "building_long": bldg[3] if bldg else None},
        "unit_scale": scale, "counts": counts, "storeys": storeys, "rooms": rooms, "elements": elements,
        "other_by_storey": other,
    }


def rnd(v, n=4):
    return None if v is None else round(v, n)


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
