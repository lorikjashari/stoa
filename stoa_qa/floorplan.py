"""Floor-plan reconstruction in image pixels: walls, rooms and openings for the 3D view of any raster plan.

The CubiCasa5K network (blueprint3d/cubicasa_model.py) gives per-pixel class maps. Its own vectoriser builds walls from
corner heatmaps and loses most walls on drawings unlike its training data (thin CAD lines, grey renders, small images).
Here:
  1. the image is cleaned (solid margins and site fill reaching the edge turn white, as on CubiCasa's paper) and
     enlarged so its long side is ~1,250 px (small images: the network sees walls at the thickness it was trained on);
  2. walls = the per-pixel wall map, united with the vectoriser's walls and with thin drawn partitions (long straight
     ink lines that meet a wall; stairs and hatching excluded); the sheet's border frame is dropped;
  3. each detector door / window is placed in the wall it sits in and cut through it; one in no wall is furniture;
  4. rooms = the space the walls enclose (doorways shut), named by the network's majority room type.
Every choice was measured on 150 CubiCasa val plans (never test) against their SVG walls, rooms, doors and windows,
as they are and shrunk to 45% (an internet-sized image); see the numbers in Params.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BLUEPRINT3D = REPO / "blueprint3d"
MAX_SIDE = 2600          # CPU time bound for one network pass (pixels on the long side after rescaling)
VECTOR_SECONDS = 60      # time budget for the network's own vectoriser; past it, the pixel wall map alone is used
ROOM_CLASSES = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room", "Bath", "Entry", "Railing",
                "Storage", "Garage", "Undefined"]


@dataclass
class Maps:
    """Network output at the working resolution (image rescaled by `factor`)."""
    factor: float
    wall: np.ndarray          # P(wall) per pixel, float32
    room: np.ndarray          # argmax room class per pixel, uint8 (ROOM_CLASSES)
    room_conf: np.ndarray     # max room-class probability, float32
    door: np.ndarray          # P(door icon)
    window: np.ndarray        # P(window icon)
    extra: dict = field(default_factory=dict)


def _model():
    if str(BLUEPRINT3D) not in sys.path:
        sys.path.insert(0, str(BLUEPRINT3D))
    import cubicasa_model as M

    return M


def read_rgb(path) -> np.ndarray:
    """Any image (PNG with alpha, palette, grey, JPEG) -> RGB uint8 on a white background."""
    from PIL import Image

    with Image.open(path) as im:
        im.load()
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        return np.asarray(im.convert("RGB")).copy()


def clean_background(rgb: np.ndarray, area: float = 0.01, depth: float = 0.02) -> np.ndarray:
    """Turn large solid areas reaching the image edge (black margins, grey site fill, coloured paper) white.

    CubiCasa plans, which the network was trained on, are drawn on white. A region is cleared only if it is big (area
    share of the image) and thick (its deepest point is `depth` x the short side from its edge): a wall that runs into
    the image edge is thin, so it stays."""
    import cv2

    h, w = rgb.shape[:2]
    img = rgb.copy()
    seen = np.zeros((h + 2, w + 2), np.uint8)
    step = max(4, min(h, w) // 60)
    seeds = [(x, 0) for x in range(0, w, step)] + [(x, h - 1) for x in range(0, w, step)] + \
            [(0, y) for y in range(0, h, step)] + [(w - 1, y) for y in range(0, h, step)]
    min_area, min_depth = area * h * w, max(6.0, depth * min(h, w))
    for x, y in seeds:
        if seen[y + 1, x + 1] or rgb[y, x].min() >= 245:
            continue
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(img.copy(), mask, (x, y), 0, (12, 12, 12), (12, 12, 12), 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (1 << 8))
        seen |= mask
        region = mask[1:-1, 1:-1].astype(bool)
        if region.sum() < min_area:
            continue
        if cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3).max() < min_depth:
            continue
        img[region] = 255
    return img


def network_maps(rgb: np.ndarray, factor: float, vector: bool = True) -> Maps:
    """Run the network on `rgb` rescaled by `factor` (capped at MAX_SIDE); maps at that working resolution."""
    import cv2
    import torch
    import torch.nn.functional as F

    M = _model()
    h0, w0 = rgb.shape[:2]
    factor = min(factor, MAX_SIDE / max(h0, w0))
    img = cv2.resize(rgb, (round(w0 * factor), round(h0 * factor)), interpolation=cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA) \
        if abs(factor - 1) > 1e-3 else rgb
    h, w = img.shape[:2]
    x = torch.from_numpy(np.moveaxis(img, -1, 0)).float().unsqueeze(0)
    x = 2 * (x / 255.0) - 1
    with torch.no_grad():
        pred = F.interpolate(M.load_model()(x), size=(h, w), mode="bilinear", align_corners=True)[0]
    rooms = torch.softmax(pred[21:33], 0)
    icons = torch.softmax(pred[33:44], 0)
    conf, lab = rooms.max(0)
    extra = {}
    if vector:  # the network's own vectoriser (corner heatmaps -> straight walls): precise on CubiCasa-like drawings
        pp = M._load_module("cc_post", "floortrans/post_prosessing.py")
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as Late

        args = ((pred[:21].numpy().copy(), rooms.numpy().copy(), icons.numpy().copy()), 0.2, [1, 2])  # it edits its inputs
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            polys, types, _, _ = pool.submit(pp.get_polygons, *args).result(timeout=VECTOR_SECONDS)
            extra["vec_walls"] = [np.asarray(p, np.float64) for p, ty in zip(polys, types) if ty["type"] == "wall"]
        except (Late, Exception):  # too slow or failed on an unusual drawing: the pixel walls still stand
            extra["vec_walls"] = []
        finally:
            pool.shutdown(wait=False)
    return Maps(factor=factor * 1.0, wall=rooms[2].numpy().astype(np.float32), room=lab.numpy().astype(np.uint8),
                room_conf=conf.numpy().astype(np.float32), door=icons[2].numpy().astype(np.float32),
                window=icons[1].numpy().astype(np.float32), extra=extra)


# ---------------------------------------------------------------- geometry helpers (working pixels)

def _odd(n: float) -> int:
    n = max(1, int(round(n)))
    return n if n % 2 else n + 1


def thickness(mask: np.ndarray) -> float:
    """Typical stroke thickness of a binary mask: median of 2 x distance-to-edge along its skeleton."""
    import cv2
    from skimage.morphology import skeletonize

    if not mask.any():
        return 0.0
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    sk = skeletonize(mask.astype(bool))
    v = dt[sk]
    return float(2 * np.median(v)) if v.size else 0.0


def _drop_small(mask: np.ndarray, min_area: float) -> np.ndarray:
    import cv2

    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def _snap(poly: np.ndarray, tol_deg: float = 8.0) -> np.ndarray:
    """Make edges within tol_deg of horizontal / vertical exactly axis-aligned (vertices move, topology stays)."""
    p = poly.astype(np.float64).copy()
    n = len(p)
    if n < 3:
        return p
    t = np.tan(np.radians(tol_deg))
    for _ in range(2):
        for i in range(n):
            a, b = p[i], p[(i + 1) % n]
            dx, dy = b[0] - a[0], b[1] - a[1]
            if abs(dy) <= t * abs(dx):      # near horizontal: share the mean y
                y = (a[1] + b[1]) / 2
                p[i][1] = p[(i + 1) % n][1] = y
            elif abs(dx) <= t * abs(dy):    # near vertical: share the mean x
                x = (a[0] + b[0]) / 2
                p[i][0] = p[(i + 1) % n][0] = x
    return p


def polygons(mask: np.ndarray, eps: float, min_area: float, snap: bool = True) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """Binary mask -> [(outer, [holes])] simplified (Douglas-Peucker eps) and optionally axis-snapped."""
    import cv2

    cs, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    if hier is None:
        return out
    hier = hier[0]
    for i, c in enumerate(cs):
        if hier[i][3] != -1 or cv2.contourArea(c) < min_area:
            continue
        outer = cv2.approxPolyDP(c, eps, True)[:, 0, :].astype(np.float64)
        holes = []
        j = hier[i][2]
        while j != -1:
            if cv2.contourArea(cs[j]) >= min_area:
                holes.append(cv2.approxPolyDP(cs[j], eps, True)[:, 0, :].astype(np.float64))
            j = hier[j][0]
        if len(outer) < 3:
            continue
        if snap:
            outer, holes = _snap(outer), [_snap(h) for h in holes if len(h) >= 3]
        out.append((outer, holes))
    return out


# ---------------------------------------------------------------- reconstruction

@dataclass
class Params:
    """Tuned on CubiCasa val (see the module docstring); the defaults are the chosen values."""
    wall_thr: float = 0.5        # P(wall) above this is wall
    eps: float = 0.3             # polygon simplification, x wall thickness
    snap: bool = True            # snap near-axis edges to exact horizontal / vertical
    close: float = 0.35          # closing of small wall gaps, x wall thickness (door openings are ~4.5 x)
    min_room: float = 1.5        # smallest room, in (wall thickness)^2 units x 10 (1.5 -> ~0.06 m2 at 0.2 m walls)
    target_thick: float = 8.0    # wall thickness (px) the network sees best: CubiCasa's median
    band: float = 0.8            # half-height of an opening cut through a wall, x wall thickness
    frame: float = 0.04          # edge band (share of the short side) where a wall piece is taken for the sheet frame
    background: bool = True      # clear solid margins / site fill reaching the image edge before the network
    walls: str = "union"         # pixel (wall map) | vector (the network's own vectoriser) | union of both
    min_factor: float = 1.0      # never shrink a plan below its own size
    scale_mode: str = "long"     # long: long side to target_long px | confidence: the scale the network is surest of walls
    target_long: float = 1250.0
    max_long: float = 1800.0     # larger plans are shrunk to this: the 2019 vectoriser takes minutes on 2,500 px
    lines: bool = True           # add thin drawn partitions (long straight ink lines meeting a wall)
    line_m: float = 2.5          # shortest such line, metres
    split: float = 1.0           # split a room when two network room types each cover this share (1 = never)


@dataclass
class Opening:
    kind: str                    # door | window
    index: int | None            # detector box index (the 3D id), None for the network's own icons
    rect: list                   # 4 corners in original pixels: the cut through the wall
    horizontal: bool             # the wall it sits in runs horizontally
    on_wall: bool                # a wall was found around it
    width: float                 # opening width along the wall, original pixels


@dataclass
class Plan:
    factor: float                # working pixels per original pixel
    thick: float                 # wall thickness, original pixels
    walls: list                  # [(outer, [holes])] original pixels
    rooms: list                  # [{"poly", "holes", "cls", "area"}] original pixels
    openings: list               # [Opening]


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    import cv2

    m = mask.astype(np.uint8)
    h, w = m.shape
    flood = np.pad(m, 1).copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    return (m | (1 - flood[1:-1, 1:-1])).astype(bool)


def ink_partitions(rgb: np.ndarray, W: np.ndarray, building: np.ndarray, t: float, length: float) -> np.ndarray:
    """Thin drawn partitions the network misses: long straight horizontal / vertical ink lines inside the building
    that meet a known wall at an end. `length` is the shortest line taken (working pixels, ~2.5 m); desks, chairs and
    text are shorter. Returned as a wall mask about 0.6 wall thicknesses thick."""
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ink = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, _odd(max(15, 3 * t)), 18)
    ink &= building.astype(np.uint8)
    L = max(15, int(length))
    half = max(1, int(round(0.3 * t)))
    cands = []  # (horizontal, x, y, w, h) of every lone long line
    for kern, horizontal in ((cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)), True),
                             (cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)), False)):
        lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kern)
        n, lab, st, _ = cv2.connectedComponentsWithStats(lines, 8)
        segs = [(i, *st[i][:4]) for i in range(1, n) if (st[i][3] if horizontal else st[i][2]) <= 1.5 * t]  # lines, not areas
        # stairs and hatching: many parallel lines close together. A partition stands alone.
        gap = length * 0.25                          # ~0.6 m
        def pos(s): return s[2] + s[4] / 2 if horizontal else s[1] + s[3] / 2           # noqa: E306
        def span(s): return (s[1], s[1] + s[3]) if horizontal else (s[2], s[2] + s[4])  # noqa: E306
        crowd = {s[0] for s in segs if sum(1 for o in segs if o is not s and abs(pos(o) - pos(s)) <= gap
                                            and min(span(o)[1], span(s)[1]) - max(span(o)[0], span(s)[0]) > 0) >= 2}
        cands += [(horizontal, x, y, w, h) for i, x, y, w, h in segs if i not in crowd]
    out = np.zeros(W.shape, bool)
    reach = _odd(1.5 * t)
    for _ in range(4):  # a partition meets a wall, or a partition already taken (office grids end on each other)
        near = cv2.dilate((W | out).astype(np.uint8), np.ones((reach, reach), np.uint8)).astype(bool)
        added = []
        for c in cands:
            horizontal, x, y, w, h = c
            ends = [(y + h // 2, x), (y + h // 2, x + w - 1)] if horizontal else [(y, x + w // 2), (y + h - 1, x + w // 2)]
            if not any(near[min(max(r, 0), W.shape[0] - 1), min(max(cc, 0), W.shape[1] - 1)] for r, cc in ends):
                continue
            added.append(c)
            if horizontal:
                yc = y + h // 2
                out[max(0, yc - half):yc + half + 1, x:x + w] = True
            else:
                xc = x + w // 2
                out[y:y + h, max(0, xc - half):xc + half + 1] = True
        if not added:
            break
        cands = [c for c in cands if c not in added]
    return out & ~W


def _drop_frame(W: np.ndarray, band: float, room: np.ndarray | None = None) -> np.ndarray:
    """Drop wall pieces lying mostly in a band along the image edge: the sheet's border frame, not the building."""
    import cv2

    if band <= 0:
        return W
    h, w = W.shape
    b = max(2, int(band * min(h, w)))
    edge = np.zeros_like(W)
    edge[:b], edge[-b:], edge[:, :b], edge[:, -b:] = True, True, True, True
    n, lab = cv2.connectedComponents(W.astype(np.uint8), connectivity=8)
    if n <= 1:
        return W
    tot = np.bincount(lab.ravel(), minlength=n)
    inb = np.bincount(lab[edge], minlength=n)
    frame = (inb >= 0.6 * np.maximum(tot, 1))
    frame[0] = False
    W = W & ~frame[lab]
    if room is None:
        return W
    # a frame joined to the building at a corner: a long wall line in the edge band with paper, not rooms, inside it
    inside = (room != 0) & (room != 2)
    t = max(2, int(0.006 * min(h, w)))
    out = W.copy()
    for axis in (0, 1):
        n_lines = h if axis == 0 else w
        cover = W.mean(axis=1) if axis == 0 else W.mean(axis=0)
        for i in list(range(b)) + list(range(n_lines - b, n_lines)):
            if cover[i] < 0.5:
                continue
            step = 1 if i < b else -1
            a0, a1 = sorted((i + step * 2 * t, i + step * 4 * t))
            a0, a1 = max(0, a0), min(n_lines, a1)
            strip = inside[a0:a1] if axis == 0 else inside[:, a0:a1]
            if strip.size and strip.mean() < 0.3:
                if axis == 0:
                    out[i] = False
                else:
                    out[:, i] = False
    return _drop_small(out, 4 * t * t)


def _place(W: np.ndarray, x0, y0, x1, y1, t: float, band: float) -> tuple[bool, bool, tuple[int, int, int, int]]:
    """Find the wall an opening box sits in: (on_wall, horizontal, cut rect x0, y0, x1, y1) in working pixels."""
    H, Wd = W.shape
    e = max(2, int(round(0.9 * t)))
    X0, Y0, X1, Y1 = max(0, int(x0) - e), max(0, int(y0) - e), min(Wd, int(np.ceil(x1)) + e), min(H, int(np.ceil(y1)) + e)
    R = W[Y0:Y1, X0:X1]
    if R.size == 0 or R.sum() < max(4, t * t):
        return False, (x1 - x0) >= (y1 - y0), (int(x0), int(y0), int(x1), int(y1))
    k = max(1, int(round(t)))
    rows = np.convolve(R.sum(1).astype(float), np.ones(k), "same")    # wall pixels in a horizontal band of height t
    cols = np.convolve(R.sum(0).astype(float), np.ones(k), "same")
    sh, sv = rows.max() / max(R.shape[1], 1), cols.max() / max(R.shape[0], 1)
    hb = band * t
    if sh >= sv:
        yc = Y0 + int(rows.argmax())
        return True, True, (int(x0), int(yc - hb), int(np.ceil(x1)), int(np.ceil(yc + hb)))
    xc = X0 + int(cols.argmax())
    return True, False, (int(xc - hb), int(y0), int(np.ceil(xc + hb)), int(np.ceil(y1)))


SCALES = (1.0, 1.5, 2.2, 3.0)
PROBE = 900  # working-pixel side of the crop each candidate scale is tried on


def choose_factor(rgb: np.ndarray, params: Params = Params()) -> float:
    """The enlargement at which the network is surest about walls (mean P(wall) over its wall pixels).

    Measured on CubiCasa val, plans shrunk to 45% (an internet-sized image): picking this way reaches wall F1 0.913
    against 0.920 for the best scale per plan, and 0.845 without enlarging. Each candidate runs on a central crop of
    about PROBE x PROBE working pixels, so trying four scales costs about one full pass."""
    h, w = rgb.shape[:2]
    if params.scale_mode == "long":  # small plans up to about target_long px; big ones down to max_long (CPU time)
        return float(min(np.clip(params.target_long / max(h, w), params.min_factor, 3.0), params.max_long / max(h, w)))
    cands = [c for c in SCALES if c >= params.min_factor and max(h, w) * c <= MAX_SIDE * 1.15]
    if len(cands) <= 1:
        return cands[0] if cands else min(1.0, MAX_SIDE / max(h, w))
    ink = rgb.min(axis=2) < 200
    ys, xs = np.nonzero(ink)
    cy, cx = (int(np.median(ys)), int(np.median(xs))) if len(ys) else (h // 2, w // 2)
    best, best_c = -1.0, cands[0]
    for c in cands:
        side = int(PROBE / c)
        y0, x0 = min(max(0, cy - side // 2), max(0, h - side)), min(max(0, cx - side // 2), max(0, w - side))
        m = network_maps(rgb[y0:y0 + side, x0:x0 + side], c, vector=False)
        sel = m.wall > params.wall_thr
        score = float(m.wall[sel].mean()) if sel.sum() >= 50 else 0.0
        if score > best:
            best, best_c = score, c
    return best_c


def reconstruct(rgb: np.ndarray, boxes: list | None = None, factor: float | None = None, maps: Maps | None = None,
                params: Params = Params()) -> Plan:
    """Walls, rooms and openings of one plan image. boxes: [(index, kind, x0, y0, x1, y1)] in original pixels, kind
    door | window (the detector boxes the answers count); without boxes the network's own door/window maps are used."""
    import cv2

    P = params
    if maps is None:
        clean = clean_background(rgb) if P.background else rgb
        maps = network_maps(clean, factor if factor is not None else choose_factor(clean, P))
    f = maps.factor
    W = maps.wall > P.wall_thr
    t = thickness(_drop_small(W, 30)) or P.target_thick
    W = _drop_small(W, max(20.0, (1.5 * t) ** 2))
    vec = maps.extra.get("vec_walls") or []
    if P.walls in ("vector", "union") and vec:
        V = np.zeros(W.shape, np.uint8)
        for p in vec:
            if len(p) >= 3:
                cv2.fillPoly(V, [np.round(p).astype(np.int32)], 1)
        W = V.astype(bool) if P.walls == "vector" else (W | V.astype(bool))
    W = _drop_frame(W, P.frame, maps.room)
    if P.close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (_odd(P.close * t * 2), _odd(P.close * t * 2)))
        W = cv2.morphologyEx(W.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    if P.lines and boxes is not None:
        doors = sorted(max(x1 - x0, y1 - y0) * f for _, kind, x0, y0, x1, y1 in boxes if kind == "door")
        ppm = (doors[len(doors) // 2] / 0.9) if len(doors) >= 2 else t / 0.2   # working pixels per metre
        clean = clean_background(rgb) if P.background else rgb
        work = cv2.resize(clean, (W.shape[1], W.shape[0]), interpolation=cv2.INTER_CUBIC) if clean.shape[:2] != W.shape else clean
        building0 = _fill_holes(W) | _fill_holes(maps.room != 0)
        W = _drop_frame(W | ink_partitions(work, W, building0, t, P.line_m * ppm), P.frame, maps.room)

    openings, cuts = [], []
    if boxes is None:  # the network's own doors and windows
        boxes = []
        for kind, prob in (("door", maps.door), ("window", maps.window)):
            n, lab, st, _ = cv2.connectedComponentsWithStats((prob > 0.5).astype(np.uint8), 8)
            for i in range(1, n):
                x, y, w, h, a = st[i]
                if a >= (0.8 * t) ** 2:
                    boxes.append((None, kind, x / f, y / f, (x + w) / f, (y + h) / f))
    for idx, kind, x0, y0, x1, y1 in boxes:
        on, horiz, (cx0, cy0, cx1, cy1) = _place(W, x0 * f, y0 * f, x1 * f, y1 * f, t, P.band)
        if on:
            cuts.append((kind, cx0, cy0, cx1, cy1))
        rect = [[cx0 / f, cy0 / f], [cx1 / f, cy0 / f], [cx1 / f, cy1 / f], [cx0 / f, cy1 / f]]
        width = ((cx1 - cx0) if horiz else (cy1 - cy0)) / f
        openings.append(Opening(kind, idx, rect, horiz, on, width))

    closed = W.copy()   # walls with every opening shut: what separates rooms
    for kind, cx0, cy0, cx1, cy1 in cuts:
        closed[max(0, cy0):cy1, max(0, cx0):cx1] = True
    cut = W.copy()      # walls with the openings cut out: what is extruded
    for kind, cx0, cy0, cx1, cy1 in cuts:
        cut[max(0, cy0):cy1, max(0, cx0):cx1] = False

    eps = max(1.0, P.eps * t)
    walls = [(o / f, [h / f for h in hs]) for o, hs in polygons(cut, eps, (0.8 * t) ** 2, P.snap)]

    building = _fill_holes(closed) | _fill_holes(maps.room != 0)
    free = building & ~cv2.dilate(closed.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    n, lab = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    rooms = []
    min_area = P.min_room * 10 * t * t
    smooth = None
    for i in range(1, n):
        m = lab == i
        a = int(m.sum())
        if a < min_area:
            continue
        cls = np.bincount(maps.room[m], minlength=len(ROOM_CLASSES))
        cls[[0, 2, 8]] = 0                      # background, wall, railing never name a room
        parts = [(m, cls)]
        big = [c for c in range(len(cls)) if cls[c] >= max(min_area, P.split * a)]
        if P.split < 1 and len(big) >= 2:       # open plan: split along the network's room types
            if smooth is None:
                smooth = _smooth_classes(maps.room, max(3, int(1.5 * t)))
            parts = []
            for c in big:
                pm = m & (smooth == c)
                k, plab = cv2.connectedComponents(pm.astype(np.uint8), connectivity=4)
                for j in range(1, k):
                    q = plab == j
                    if q.sum() >= min_area:
                        parts.append((q, np.bincount(maps.room[q], minlength=len(ROOM_CLASSES))))
            if not parts:
                parts = [(m, cls)]
        for q, qc in parts:
            qc = qc.copy()
            qc[[0, 2, 8]] = 0
            name = ROOM_CLASSES[int(qc.argmax())] if qc.sum() else "Undefined"
            for o, hs in polygons(q, max(1.0, 0.4 * t), min_area / 2, P.snap)[:1]:
                rooms.append({"poly": o / f, "holes": [h / f for h in hs], "cls": name, "area": int(q.sum()) / f / f})
    return Plan(f, t / f, walls, rooms, openings)


def _smooth_classes(room: np.ndarray, k: int) -> np.ndarray:
    """Room-type map with speckles removed: each pixel takes the type most present in a k x k window."""
    import cv2

    best = np.full(room.shape, -1.0, np.float32)
    out = np.zeros(room.shape, np.uint8)
    for c in range(len(ROOM_CLASSES)):
        if c in (0, 2, 8) or not (room == c).any():
            continue
        s = cv2.blur((room == c).astype(np.float32), (k, k))
        better = s > best
        out[better], best[better] = c, s[better]
    return out


# ---------------------------------------------------------------- persistence (one reconstruction per image)

def to_json(plan: Plan) -> dict:
    L = lambda a: np.asarray(a, float).round(2).tolist()  # noqa: E731
    return {"factor": plan.factor, "thick": plan.thick, "walls": [[L(o), [L(h) for h in hs]] for o, hs in plan.walls],
            "rooms": [{"poly": L(r["poly"]), "holes": [L(h) for h in r["holes"]], "cls": r["cls"], "area": r["area"]}
                      for r in plan.rooms],
            "openings": [{"kind": o.kind, "index": o.index, "rect": L(o.rect), "horizontal": o.horizontal,
                          "on_wall": o.on_wall, "width": o.width} for o in plan.openings]}


def from_json(d: dict) -> Plan:
    A = lambda a: np.asarray(a, float)  # noqa: E731
    return Plan(d["factor"], d["thick"], [(A(o), [A(h) for h in hs]) for o, hs in d["walls"]],
                [{"poly": A(r["poly"]), "holes": [A(h) for h in r["holes"]], "cls": r["cls"], "area": r["area"]} for r in d["rooms"]],
                [Opening(o["kind"], o["index"], o["rect"], o["horizontal"], o["on_wall"], o["width"]) for o in d["openings"]])
