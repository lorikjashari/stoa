"""Turn a raster floor plan (PNG/JPG scan, e.g. Stoa cubicasa5k) into a 3D scene, approximately.

    .venv/Scripts/python blueprint3d/raster_to_scene.py <plan.png> [...] [-o out_dir] [--json] [--debug dir]

App contract (archive scans and CubiCasa alike): build(path, out_dir) -> scene dict, or raises NotAPlan(reason).
CLI with --json prints one JSON line per image on stdout:
  {"ok": true,  "file", "scene": "<out>/<stem>.json", "counts", "pixels_per_m", "scale_from", "region_px", ...}
  {"ok": false, "file", "kind": "not_a_plan" | "too_little", "reason": "<one sentence>"}  (+ <out>/<stem>.reject.json)
  {"ok": false, "file", "kind": "error", "error": "..."}
Exit code: 0 all built, 2 at least one image is not a plan, 1 at least one error.
Archive scans (wikimedia, finna, europeana, habs): blueprint3d/archive_plan.py finds the plan on the sheet first.

Pipeline (classical computer vision, no training):
  1. walls     = thick dark strokes: dark mask -> (fill double-line/hatched walls) -> morphological opening
  2. openings  = gaps that close when the wall mask is closed with a door-width kernel
  3. rooms     = connected free space inside the building once openings are closed
  4. doors vs windows: a gap with a swing arc beside it is a door; a gap on the exterior without one is a window
  5. scale     = door width (0.9 m) or wall thickness, so the 3D model has plausible metres. Approximate.
Output uses the same scene.json schema as dxf_to_scene.py, so the same viewer renders it.
"""
import argparse, json, math, os, re, sys

import numpy as np

try:
    import cv2  # the OpenCV engine and fresh network predictions need it; cached predictions do not
except ImportError:
    cv2 = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = {"storey_height": 2.6, "door_head": 2.1, "window_sill": 0.9, "window_head": 2.1, "fixture_height": 0.9}
DOOR_M, WALL_M = 0.9, 0.2  # typical clear door width / wall thickness used to infer scale


class NotAPlan(Exception):
    """The image holds no usable floor plan (elevation, section, detail, site map, text, photo...).
    .reason is one sentence for the user; .kind is 'not_a_plan' or 'too_little'."""

    def __init__(self, reason, kind="not_a_plan", details=None):
        super().__init__(reason)
        self.reason, self.kind, self.details = reason, kind, details or {}


# ---------------------------------------------------------------- masks

def dark_mask(gray, style):
    if style == "colorful":
        return (gray < 140).astype(np.uint8)
    # scans and prints: Otsu, but never call mid-grey paper "ink"
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (gray < min(t, 160)).astype(np.uint8)


def stroke_widths(mask):
    """Widths of strokes at their ridge (2 x distance-transform maxima)."""
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt > 0)
    return dt, 2 * dt[ridge]


def wall_mask(dark, style):
    h, w = dark.shape
    m = dark.copy()
    if style == "high_quality_architectural":
        # walls are drawn as two parallel lines with hatching: fill the band between them first
        k = max(3, round(min(h, w) / 180))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    dt, widths = stroke_widths(m)
    widths = widths[widths >= 1]
    if widths.size == 0:
        return np.zeros_like(dark), np.zeros_like(dark), 0
    # most ink is thin linework (text, furniture, symbols); walls are clearly thicker than that
    line_w = float(np.median(widths))
    k = max(3, int(round(line_w * 1.6)) + 1)
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    _, wall_widths = stroke_widths(opened)
    thick = float(np.median(wall_widths)) if wall_widths.size else k
    # walls are long: drop compact blobs (filled symbols, text clusters, heater icons)
    n, lab, st, _ = cv2.connectedComponentsWithStats(opened, 8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if max(bw, bh) >= 8 * thick and a >= 6 * thick * thick:
            keep[i] = True
    # `opened` keeps small solid pieces (window mullions) that separate adjacent openings
    return keep[lab].astype(np.uint8), opened, max(2, int(round(thick)))


def swing_arc(ink, x, y, bw, bh, horizontal, samples=24):
    """Best fraction of a quarter circle (radius = gap length, hinged at either gap end, on either side) covered by ink."""
    h, w = ink.shape
    L = bw if horizontal else bh
    best = 0.0
    if horizontal:
        cy_opts = [(y, -1), (y + bh, 1)]  # (wall face, side direction)
        hinges = [x, x + bw]
    else:
        cy_opts = [(x, -1), (x + bw, 1)]
        hinges = [y, y + bh]
    for face, side in cy_opts:
        for hg in hinges:
          for rf in (0.7, 0.8, 0.9, 1.0, 1.15):  # the frame is part of the gap, so the leaf can be shorter
            R = L * rf
            toward = 1 if hg == hinges[0] else -1  # the arc sweeps from the leaf back across the gap
            hit = 0
            for s in range(samples):
                t = (s + 0.5) / samples * math.pi / 2
                along, out = math.cos(t) * R * toward, math.sin(t) * R * side
                px, py = (hg + along, face + out) if horizontal else (face + out, hg + along)
                if 0 <= int(py) < h and 0 <= int(px) < w and ink[int(py), int(px)]:
                    hit += 1
            best = max(best, hit / samples)
    return best


# ---------------------------------------------------------------- geometry

def polys(mask, eps):
    """Outer contours with their holes, simplified, as [[x, y], ...] lists."""
    cs, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    if hier is None:
        return out
    for i, c in enumerate(cs):
        if hier[0][i][3] != -1:
            continue
        outer = cv2.approxPolyDP(c, eps, True)[:, 0, :].tolist()
        holes, j = [], hier[0][i][2]
        while j != -1:
            hc = cv2.approxPolyDP(cs[j], eps, True)[:, 0, :].tolist()
            if len(hc) >= 3 and cv2.contourArea(cs[j]) > 4 * eps * eps:
                holes.append(hc)
            j = hier[0][j][0]
        if len(outer) >= 3:
            out.append((outer, holes))
    return out


def analyse(path, style=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read {path}")
    style = style or re.sub(r"-\d+$", "", os.path.splitext(os.path.basename(path))[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = dark_mask(gray, style)
    walls, opened, thick = wall_mask(dark, style)
    h, w = walls.shape
    if thick == 0 or walls.sum() == 0:
        return {"img": img, "walls": walls, "thick": 0, "rooms": [], "gaps": [], "outside": np.ones_like(walls), "style": style}

    # close door/window gaps along wall axes; window runs can be long (~3 m ≈ 15 wall thicknesses)
    g = int(max(9, thick * 16))
    solid = cv2.bitwise_or(walls, cv2.bitwise_and(opened, cv2.dilate(walls, np.ones((3, 3), np.uint8), iterations=int(g))))
    closed = solid.copy()
    for kern in (cv2.getStructuringElement(cv2.MORPH_RECT, (g, 1)), cv2.getStructuringElement(cv2.MORPH_RECT, (1, g))):
        closed = cv2.bitwise_or(closed, cv2.morphologyEx(solid, cv2.MORPH_CLOSE, kern))

    free = (1 - closed).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(free, 4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    outside = np.isin(lab, list(border)).astype(np.uint8)
    min_room = (thick * 12) ** 2
    rooms = [i for i in range(1, n) if i not in border and st[i][4] >= min_room]

    # gaps = what closing added between solid wall pieces
    added = cv2.subtract(closed, solid)
    added = cv2.morphologyEx(added, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(2, thick // 2),) * 2))
    gn, glab, gst, _ = cv2.connectedComponentsWithStats(added, 8)
    ink = cv2.dilate(cv2.subtract(dark, walls), np.ones((3, 3), np.uint8), iterations=max(1, thick // 4))
    out_d = cv2.dilate(outside, np.ones((3, 3), np.uint8), iterations=max(2, thick // 2))
    gaps = []
    for i in range(1, gn):
        x, y, bw, bh, a = gst[i]
        long_, short = max(bw, bh), min(bw, bh)
        if long_ < thick * 2 or short > thick * 2.5 or short < thick * 0.4 or a < 0.5 * long_ * short:
            continue
        horizontal = bw >= bh
        arc = swing_arc(ink, x, y, bw, bh, horizontal)
        exterior = bool(out_d[y:y + bh, x:x + bw].any())
        if arc >= 0.5:
            kind = "door"
        elif exterior:
            kind = "window"
        else:
            kind = "opening"
        gaps.append({"bbox": (int(x), int(y), int(bw), int(bh)), "horizontal": bool(horizontal), "arc": float(arc),
                     "exterior": exterior, "kind": kind})
    return {"img": img, "walls": walls, "thick": thick, "lab": lab, "rooms": rooms, "gaps": gaps,
            "outside": outside, "style": style, "stats": st}


# ---------------------------------------------------------------- scene

def build_scene(path, heights=DEFAULTS, debug_dir=None):
    A = analyse(path)
    h, w = A["walls"].shape
    thick = A["thick"]
    doors = [gp for gp in A["gaps"] if gp["kind"] == "door"]
    door_px = float(np.median([max(gp["bbox"][2], gp["bbox"][3]) for gp in doors])) if len(doors) >= 2 else None
    ppm = door_px / DOOR_M if door_px else (thick / WALL_M if thick else 50.0)  # pixels per metre
    scale_src = "median door width = 0.9 m" if door_px else "wall thickness = 0.2 m"
    eps = max(1.0, thick * 0.35)
    P = lambda p: [[round(x / ppm, 3), round((h - y) / ppm, 3)] for x, y in p]  # image px -> metres, y up

    walls = [{"id": f"wall:{i}", "layer": "RASTER_WALL", "poly": P(o), "holes": [P(hh) for hh in hs]}
             for i, (o, hs) in enumerate(polys(A["walls"], eps)) if cv2.contourArea(np.array(o)) > thick * thick * 4]
    rooms = []
    for k, i in enumerate(A["rooms"]):
        m = (A["lab"] == i).astype(np.uint8)
        pp = polys(m, eps)
        if not pp:
            continue
        o, hs = max(pp, key=lambda t: cv2.contourArea(np.array(t[0])))
        area_m2 = float(m.sum()) / ppm ** 2
        M = cv2.moments(m, True)
        cx, cy = (M["m10"] / M["m00"], M["m01"] / M["m00"]) if M["m00"] else (o[0][0], o[0][1])
        rooms.append({"id": f"room:{k + 1}", "layer": "RASTER_ROOM", "number": str(k + 1), "label": None, "kind": "other",
                      "outdoor": False, "area_m2": round(area_m2, 2), "printed_area": None, "unit": None,
                      "poly": P(o), "holes": [P(hh) for hh in hs], "label_at": P([[cx, cy]])[0]})
    win, drs = [], []
    for i, gp in enumerate(A["gaps"]):
        x, y, bw, bh = gp["bbox"]
        rect = [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]
        if gp["kind"] == "window":
            win.append({"id": f"window:{i}", "layer": "RASTER_GAP", "poly": P(rect)})
        elif gp["kind"] == "door":
            # leaf: a thin panel standing open at one end of the gap, on the arc side
            t = max(2, thick // 3)
            leaf = [[x, y], [x + t, y], [x + t, y + bw], [x, y + bw]] if gp["horizontal"] else [[x, y], [x + bh, y], [x + bh, y + t], [x, y + t]]
            drs.append({"id": f"door:{i}", "layer": "RASTER_GAP", "leaf": P(leaf), "lintel": P(rect),
                        "width_m": round(max(bw, bh) / ppm, 2)})
    if debug_dir:
        write_debug(A, os.path.join(debug_dir, os.path.splitext(os.path.basename(path))[0] + "_debug.jpg"))
    allpts = [q for g in (walls, rooms) for el in g for q in el["poly"]] or [[0, 0], [w / ppm, h / ppm]]
    xs, ys = [q[0] for q in allpts], [q[1] for q in allpts]
    rel = os.path.relpath(path).replace("\\", "/")
    m = re.search(r"data/.*", rel)
    return {
        "schema": "blueprint3d/scene@1",
        "source": {"file": m.group(0) if m else os.path.basename(path), "format": "raster", "style": A["style"],
                   "units": "m", "pixels_per_m": round(ppm, 2), "scale_from": scale_src},
        "assumptions": {**heights, "note": f"Approximate: geometry traced from a raster image; scale from {scale_src}; heights assumed."},
        "bbox": [min(xs), min(ys), max(xs), max(ys)],
        "counts": {"rooms": len(rooms), "walls": len(walls), "doors": len(drs), "windows": len(win), "units": 0, "fixtures": 0},
        "units": [], "rooms": rooms, "walls": walls, "doors": drs, "windows": win, "fixtures": [],
    }


# ---------------------------------------------------------------- engine 2: pretrained CubiCasa5K model

MODEL_ROOM_KIND = {"Kitchen": "kitchen", "Living Room": "living", "Bed Room": "bedroom", "Bath": "bath",
                   "Entry": "corridor", "Storage": "storage", "Outdoor": "outdoor", "Garage": "other", "Undefined": "other"}
FIXTURE_CLASSES = {"Toilet", "Sink", "Bathtub", "Sauna Bench", "Closet", "Electrical Appliance", "Fire Place", "Chimney"}


def build_scene_model(path, heights=DEFAULTS):
    """Scene from the CubiCasa5K network: walls, typed rooms, doors, windows and fixtures (cached predictions)."""
    from eval_cubicasa_model import detect
    rel = os.path.relpath(path, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "junction-dataset", "junction-dataset")).replace("\\", "/")
    D = detect(rel)
    from PIL import Image
    with Image.open(path) as im:
        h = im.height
    side = lambda p: sorted([math.dist(p[0], p[1]), math.dist(p[1], p[2])]) if len(p) >= 3 else [0, 0]
    doors_px = [side(i["poly"])[1] for i in D["icons"] if i["class"] == "Door"]
    walls_px = [side(wl["poly"])[0] for wl in D["walls"]]
    if len(doors_px) >= 2:
        ppm, scale_src = float(np.median(doors_px)) / DOOR_M, "median detected door width = 0.9 m"
    elif walls_px:
        ppm, scale_src = float(np.median(walls_px)) / WALL_M, "median wall thickness = 0.2 m"
    else:
        ppm, scale_src = 50.0, "default 50 px/m"
    P = lambda p: [[round(x / ppm, 3), round((h - y) / ppm, 3)] for x, y in p]
    walls = [{"id": f"wall:{i}", "layer": "MODEL_WALL", "poly": P(wl["poly"])} for i, wl in enumerate(D["walls"]) if len(wl["poly"]) >= 3]
    rooms, k = [], 0
    for r in D["rooms"]:
        kind = MODEL_ROOM_KIND.get(r["class"])
        if not kind:
            continue
        k += 1
        poly = r["poly"]
        cx, cy = sum(x for x, _ in poly) / len(poly), sum(y for _, y in poly) / len(poly)
        area = abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]))) / 2 / ppm ** 2
        rooms.append({"id": f"room:{k}", "layer": "MODEL_ROOM", "number": str(k), "label": r["class"], "kind": kind,
                      "outdoor": kind == "outdoor", "area_m2": round(area, 2), "printed_area": None, "unit": None,
                      "poly": P(poly), "holes": [P(hh) for hh in r.get("holes", [])], "label_at": P([[cx, cy]])[0]})
    windows, doors, fixtures = [], [], []
    for i, ic in enumerate(D["icons"]):
        p = ic["poly"]
        if len(p) < 3:
            continue
        if ic["class"] == "Window":
            windows.append({"id": f"window:{i}", "layer": "MODEL_WINDOW", "poly": P(p)})
        elif ic["class"] == "Door":
            xs, ys = [q[0] for q in p], [q[1] for q in p]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            t = max(2.0, 0.05 * ppm)
            leaf = [[x0, y0], [x0 + t, y0], [x0 + t, y0 + (x1 - x0)], [x0, y0 + (x1 - x0)]] if x1 - x0 >= y1 - y0 \
                else [[x0, y0], [x0 + (y1 - y0), y0], [x0 + (y1 - y0), y0 + t], [x0, y0 + t]]
            doors.append({"id": f"door:{i}", "layer": "MODEL_DOOR", "leaf": P(leaf), "lintel": P(p),
                          "width_m": round(max(x1 - x0, y1 - y0) / ppm, 2)})
        elif ic["class"] in FIXTURE_CLASSES:
            fixtures.append({"id": f"fixture:{i}", "layer": "MODEL_ICON", "label": ic["class"], "poly": P(p)})
    allpts = [q for g in (walls, rooms) for el in g for q in el["poly"]] or [[0, 0], [1, 1]]
    xs, ys = [q[0] for q in allpts], [q[1] for q in allpts]
    c = D["counts"]
    return {
        "schema": "blueprint3d/scene@1",
        "source": {"file": rel, "format": "raster", "engine": "cubicasa5k-model", "units": "m",
                   "pixels_per_m": round(ppm, 2), "scale_from": scale_src},
        "assumptions": {**heights, "note": f"Approximate: CubiCasa5K network on a raster image; scale from {scale_src}; heights assumed."},
        "bbox": [min(xs), min(ys), max(xs), max(ys)],
        "counts": {"rooms": len(rooms), "walls": len(walls), "doors": len(doors), "windows": len(windows), "units": 0,
                   "fixtures": len(fixtures), "saunas": c["saunas"]},
        "units": [], "rooms": rooms, "walls": walls, "doors": doors, "windows": windows, "fixtures": fixtures,
    }


# ---------------------------------------------------------------- engine 3: archive scans (wikimedia, finna, europeana, habs)

ARCHIVE_SOURCES = ("wikimedia", "finna", "europeana", "habs")
ARCHIVE_HEIGHTS = {"storey_height": 3.2, "door_head": 2.4, "window_sill": 1.0, "window_head": 2.4, "fixture_height": 0.9}
ARCHIVE_WALL_M = 0.45      # historic masonry; used for scale only when no doors are found
ARCHIVE_SPAN_M = (6, 150)  # a traced building's long side must land in this range, or the scale is clamped to it
MIN_WALL_COVERAGE = 0.5    # share of the network's wall pixels that the vectorised walls must explain
ARCHIVE_NOTE = ("Approximate, traced from an archive scan: the CubiCasa5K network (trained on modern Finnish "
                "apartment plans) found the walls, rooms and openings; scale from {scale}; heights assumed. "
                "Not a survey.")


def _archive_scale(walls_px, doors_px, span_px):
    """Pixels per metre on the network crop, and where it came from."""
    if len(doors_px) >= 2:
        d = np.asarray(doors_px, float)
        med = float(np.median(d))
        core = d[(d >= 0.6 * med) & (d <= 1.6 * med)]  # drop double doors, gates and misreads
        if len(core) >= max(2, 0.6 * len(d)):
            ppm, src = float(np.median(core)) / DOOR_M, f"median width of {len(core)} detected doors = {DOOR_M} m"
            if ARCHIVE_SPAN_M[0] <= span_px / ppm <= ARCHIVE_SPAN_M[1]:
                return ppm, src
    if walls_px:
        ppm = float(np.median(walls_px)) / ARCHIVE_WALL_M
        src = f"median wall thickness = {ARCHIVE_WALL_M} m (assumed for historic masonry; no reliable doors)"
    else:
        ppm, src = span_px / 20.0, "assumed 20 m building length (no doors or walls to measure)"
    lo, hi = ARCHIVE_SPAN_M
    if span_px / ppm > hi:
        ppm, src = span_px / hi, f"clamped: building assumed {hi} m long (the {src.split(' (')[0]} gave an implausible size)"
    elif span_px / ppm < lo:
        ppm, src = span_px / lo, f"clamped: building assumed {lo} m long (the {src.split(' (')[0]} gave an implausible size)"
    return ppm, src


def build_scene_archive(path, heights=ARCHIVE_HEIGHTS):
    """Scene from an archive scan, or NotAPlan(reason) when the sheet holds no usable floor plan."""
    import archive_plan as AP
    import cubicasa_model as M
    R = AP.analyse_sheet(path)
    if R["best"] is None:
        raise NotAPlan(R["reason"], kind="not_a_plan", details={"views": len(R["views"])})
    heat, rooms_p, icons_p, crop = R["pred"]
    v = R["views"][R["best"]]
    polygons, types, _, _ = M.vectorize(heat, rooms_p.copy(), icons_p)  # walls + openings; rooms below
    h_net = crop.shape[0]
    side = lambda p: sorted([math.dist(p[0], p[1]), math.dist(p[1], p[2])]) if len(p) >= 3 else [0, 0]
    walls_px, doors_px, wall_polys, icon_polys = [], [], [], []
    for poly, ty in zip(polygons, types):
        pts = [[float(x), float(y)] for x, y in poly]
        if ty["type"] == "icon":
            icon_polys.append((M.ICON_CLASSES[ty["class"]], pts))
            if M.ICON_CLASSES[ty["class"]] == "Door":
                doors_px.append(side(pts)[1])
        else:
            wall_polys.append(pts)
            walls_px.append(side(pts)[0])
    if len(wall_polys) < 4:
        raise NotAPlan("The sheet looked like a floor plan, but too few walls could be traced from it to build a model.",
                       kind="too_little")
    # how much of the network's wall segmentation the vectorised walls explain: low = walls traced in pieces only
    seg_wall = rooms_p.argmax(0) == AP.WALL
    traced = np.zeros(seg_wall.shape, np.uint8)
    for p in wall_polys:
        cv2.fillPoly(traced, [np.round(np.array(p)).astype(np.int32)], 1)
    traced = cv2.dilate(traced, np.ones((3, 3), np.uint8), iterations=3)
    coverage = float(traced[seg_wall].mean()) if seg_wall.any() else 0.0
    if coverage < MIN_WALL_COVERAGE:
        raise NotAPlan(f"The sheet looks like a floor plan, but only {coverage:.0%} of its walls could be traced "
                       "as straight wall segments, too few for a faithful model.", kind="too_little")
    allx = [q[0] for p in wall_polys for q in p]
    ally = [q[1] for p in wall_polys for q in p]
    span_px = max(max(allx) - min(allx), max(ally) - min(ally))
    from shapely.geometry import MultiPoint
    hull = MultiPoint([(x, y) for x, y in zip(allx, ally)]).convex_hull
    ppm, scale_src = _archive_scale(walls_px, doors_px, span_px)
    P = lambda p: [[round(x / ppm, 3), round((h_net - y) / ppm, 3)] for x, y in p]
    walls = [{"id": f"wall:{i}", "layer": "MODEL_WALL", "poly": P(p)} for i, p in enumerate(wall_polys)]
    from shapely.geometry import Polygon
    # rooms straight from the room segmentation (blobs the predicted walls enclose): on archive scans this follows
    # the drawing far better than the repo's junction-grid rooms, which need clean CubiCasa-style corners
    am = rooms_p.argmax(0)
    inter = np.isin(am, AP.INTERIOR).astype(np.uint8)
    near_wall = cv2.dilate((am == AP.WALL).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inter, 4)
    rooms, k = [], 0
    for i in range(1, n):
        if st[i][4] / ppm ** 2 < 2.0:  # slivers under 2 m2
            continue
        m = (lab == i).astype(np.uint8)
        ring = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=2) - m
        if not ring.any() or near_wall[ring > 0].mean() < 0.6:  # not enclosed by walls: outside, courtyard, margin
            continue
        cs, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        c0 = max(cs, key=cv2.contourArea)
        outer = cv2.approxPolyDP(c0, max(1.0, 0.12 * ppm), True)[:, 0, :].astype(float)
        if len(outer) < 3:
            continue
        g = Polygon(outer).buffer(0)
        if g.is_empty or g.geom_type != "Polygon" or g.intersection(hull).area < 0.8 * g.area:
            continue  # outside the traced building: title lettering, legends, neighbouring buildings
        k += 1
        c = g.representative_point()
        # the network's room types (kitchen, sauna...) are Finnish-apartment priors: not shown for archive scans
        rooms.append({"id": f"room:{k}", "layer": "MODEL_ROOM", "number": str(k), "label": None, "kind": "other",
                      "outdoor": False, "area_m2": round(g.area / ppm ** 2, 1), "printed_area": None, "unit": None,
                      "poly": P([[float(x), float(y)] for x, y in g.exterior.coords[:-1]]), "holes": [],
                      "label_at": P([[c.x, c.y]])[0]})
    windows, doors = [], []
    for i, (cls, p) in enumerate(icon_polys):
        if len(p) < 3:
            continue
        if cls == "Window":
            windows.append({"id": f"window:{i}", "layer": "MODEL_WINDOW", "poly": P(p)})
        elif cls == "Door":
            xs, ys = [q[0] for q in p], [q[1] for q in p]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            t = max(2.0, 0.05 * ppm)
            leaf = [[x0, y0], [x0 + t, y0], [x0 + t, y0 + (x1 - x0)], [x0, y0 + (x1 - x0)]] if x1 - x0 >= y1 - y0 \
                else [[x0, y0], [x0 + (y1 - y0), y0], [x0 + (y1 - y0), y0 + t], [x0, y0 + t]]
            doors.append({"id": f"door:{i}", "layer": "MODEL_DOOR", "leaf": P(leaf), "lintel": P(p),
                          "width_m": round(max(x1 - x0, y1 - y0) / ppm, 2)})
    pts = [q for g in (walls, rooms) for el in g for q in el["poly"]]
    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
    rel = os.path.relpath(path).replace("\\", "/")
    m = re.search(r"data/.*", rel)
    others = R["plans"] - 1
    note = ARCHIVE_NOTE.format(scale=scale_src)
    if others > 0:
        note += f" The sheet holds {others + 1} plans; this model shows the clearest one."
    # crop region on the original scan + network px -> original px factor, so the app can outline the traced area
    k_orig = R["orig_wh"][0] / R["rgb_hw"][1]
    return {
        "schema": "blueprint3d/scene@1",
        "source": {"file": m.group(0) if m else os.path.basename(path), "format": "raster", "engine": "cubicasa5k-model",
                   "style": "archive", "units": "m", "pixels_per_m": round(ppm, 2), "scale_from": scale_src,
                   "region_px": R["region"], "image_px_per_net_px": round(k_orig / v["net_scale"], 4),
                   "plans_on_sheet": R["plans"], "plan_check": v["why"], "wall_trace_coverage": round(coverage, 2)},
        "assumptions": {**heights, "note": note},
        "bbox": [min(xs), min(ys), max(xs), max(ys)],
        "counts": {"rooms": len(rooms), "walls": len(walls), "doors": len(doors), "windows": len(windows), "units": 0,
                   "fixtures": 0},
        "units": [], "rooms": rooms, "walls": walls, "doors": doors, "windows": windows, "fixtures": [],
    }


def build(path, out_dir=None):
    """The one entry point for the app: image -> scene dict (also written to <out_dir>/<stem>.json).

    CubiCasa5K plans use the cached/fresh network prediction exactly as before (build_scene_model). Archive scans
    (wikimedia, finna, europeana, habs, or any other raster) go through the floor-plan check first and raise
    NotAPlan(reason) when the sheet is an elevation, section, detail, site map, text or photo."""
    scene = build_scene_archive(path) if not re.search(r"cubicasa5k", path.replace("\\", "/")) else build_scene_model(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + ".json"), "w", encoding="utf-8") as f:
            json.dump(scene, f, ensure_ascii=False, separators=(",", ":"))
    return scene


def write_debug(A, out):
    img = A["img"].copy()
    ov = img.copy()
    rng = np.random.default_rng(1)
    if "lab" in A:
        for i in A["rooms"]:
            ov[A["lab"] == i] = rng.integers(80, 230, 3)
    ov[A["walls"] > 0] = (40, 40, 200)
    img = cv2.addWeighted(img, 0.45, ov, 0.55, 0)
    col = {"door": (0, 160, 255), "window": (255, 150, 0), "opening": (160, 160, 160)}
    for gp in A["gaps"]:
        x, y, bw, bh = gp["bbox"]
        cv2.rectangle(img, (x - 2, y - 2), (x + bw + 2, y + bh + 2), col[gp["kind"]], 3)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 80])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("-o", "--out", default=os.path.join(os.path.dirname(__file__), "viewer", "scenes"))
    ap.add_argument("--debug", help="write a debug overlay per image into this folder (opencv engine)")
    ap.add_argument("--engine", choices=["model", "opencv"], default="model",
                    help="model = pretrained CubiCasa5K network (default); opencv = hand-tuned baseline")
    ap.add_argument("--json", action="store_true", help="print exactly one JSON line per image on stdout")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    index_path = os.path.join(a.out, "index.json")
    index = json.load(open(index_path, encoding="utf-8")) if os.path.exists(index_path) else []
    status = 0
    for p in a.images:
        name = os.path.splitext(os.path.basename(p))[0]
        scene_path = os.path.join(a.out, name + ".json")
        reject_path = os.path.join(a.out, name + ".reject.json")
        try:
            if a.engine == "opencv":
                scene = build_scene(p, DEFAULTS, a.debug)
                with open(scene_path, "w", encoding="utf-8") as f:
                    json.dump(scene, f, ensure_ascii=False, separators=(",", ":"))
            else:
                scene = build(p, a.out)
        except NotAPlan as e:
            status = max(status, 2)
            out = {"ok": False, "file": p, "kind": e.kind, "reason": e.reason}
            with open(reject_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False)
            if os.path.exists(scene_path):  # a scene from an older builder would contradict the verdict
                os.remove(scene_path)
            index = [x for x in index if x["name"] != name]
            print(json.dumps(out, ensure_ascii=False) if a.json else f"{name}: no 3D - {e.reason}", flush=True)
            continue
        except Exception as e:  # unreadable image, missing weights...
            status = 1
            out = {"ok": False, "file": p, "kind": "error", "error": f"{type(e).__name__}: {e}"[:500]}
            print(json.dumps(out, ensure_ascii=False) if a.json else f"{name}: error - {out['error']}", flush=True)
            continue
        if os.path.exists(reject_path):
            os.remove(reject_path)
        index = [x for x in index if x["name"] != name] + [{"name": name, "source": scene["source"]["file"], "counts": scene["counts"]}]
        c, s = scene["counts"], scene["source"]
        if a.json:
            print(json.dumps({"ok": True, "file": p, "scene": scene_path, "counts": c, "pixels_per_m": s["pixels_per_m"],
                              "scale_from": s["scale_from"], "region_px": s.get("region_px"),
                              "plans_on_sheet": s.get("plans_on_sheet"), "note": scene["assumptions"]["note"]},
                             ensure_ascii=False), flush=True)
        else:
            print(f"{name}: {c['rooms']} rooms, {c['walls']} walls, {c['doors']} doors, {c['windows']} windows "
                  f"({s['pixels_per_m']} px/m from {s['scale_from']})", flush=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(sorted(index, key=lambda x: x["name"]), f, ensure_ascii=False, indent=1)
    return status


if __name__ == "__main__":
    sys.exit(main())
