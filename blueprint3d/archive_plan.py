"""Archive scans (wikimedia, finna, europeana, habs) -> is this a floor plan, and where on the sheet is it?

The CubiCasa5K network was trained on clean Finnish apartment plans. Archive sheets are old hand drawings, colour
scans, 90 MP HABS sheets and multi-view sheets (plan + elevation + section + title block), and most of them are not
plans at all. This module:
  1. loads the scan at a sane working size (JPEG draft decoding, so a 90 MP sheet never decodes in full),
  2. finds the separate drawings on the sheet (ink islands after removing the frame and title-block rules),
  3. runs the network on each candidate at a resolution where its walls look like CubiCasa walls,
  4. scores each candidate as a floor plan from the network's own outputs plus cheap image features,
and either returns the best plan region with its network prediction or a reason why the sheet is not a plan.
No LLM calls. Measured on a hand-labelled sample: _analysis/archive_eval.py.
"""
import os, re, sys, unicodedata

import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
Image.MAX_IMAGE_PIXELS = None  # HABS sheets are huge, and they are trusted dataset files

INTERIOR = [3, 4, 5, 6, 7, 9, 10, 11]  # Kitchen, Living, Bed, Bath, Entry, Storage, Garage, Undefined
WALL, BG, OUTDOOR = 2, 0, 1


# ---------------------------------------------------------------- loading

def load_rgb(path, max_side=3000):
    """RGB uint8 array with long side <= max_side. JPEG draft mode decodes big scans at 1/2..1/8 directly."""
    with Image.open(path) as im:
        if im.format == "JPEG" and max(im.size) > max_side:
            im.draft("RGB", (max_side, max_side))
        im = im.convert("RGB")
        s = min(1.0, max_side / max(im.size))
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        a = np.asarray(im)
    # blueprints / negatives: light lines on a dark ground -> dark lines on light paper (what the network knows)
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    if np.median(g) < 110 and (g > np.median(g) + 60).mean() < 0.25:
        a = 255 - a
    return a


def ink_mask(gray):
    """Ink = clearly darker than the local paper. Works on yellowed, stained and faint scans."""
    h, w = gray.shape
    k = max(15, (min(h, w) // 25) | 1)
    paper = cv2.medianBlur(cv2.resize(gray, (w // 4 or 1, h // 4 or 1), interpolation=cv2.INTER_AREA), k // 4 | 1)
    paper = cv2.resize(paper, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.int16)
    diff = paper - gray.astype(np.int16)
    t = max(25, int(np.percentile(diff, 99) * 0.35)) if diff.size else 25
    return (diff > t).astype(np.uint8)


# ---------------------------------------------------------------- views on the sheet

def remove_rules(ink, frac=0.3):
    """Drop long straight lines (sheet frame, title-block rules, crop marks) so drawings separate into islands.
    Tolerates a slightly rotated scan: lines are thickened across before the long opening."""
    h, w = ink.shape
    out = ink.copy()
    for k, fat in (((max(40, int(w * frac)), 1), (1, 5)), ((1, max(40, int(h * frac))), (5, 1))):
        fatter = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, fat[::-1]))
        lines = cv2.morphologyEx(fatter, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
        out[cv2.dilate(lines, np.ones((5, 5), np.uint8)) > 0] = 0
    return out


def _xy_cut(ink, box, min_gap, out, depth=0, parent=-1):
    """Recursive XY-cut: split a region at white gaps (rows/columns with no ink) of at least min_gap px."""
    x0, y0, x1, y1 = box
    sub = ink[y0:y1, x0:x1]
    ys, xs = np.nonzero(sub)
    if len(xs) == 0:
        return
    x0, x1, y0, y1 = x0 + xs.min(), x0 + xs.max() + 1, y0 + ys.min(), y0 + ys.max() + 1  # trim to ink
    me = len(out)
    out.append({"box": (int(x0), int(y0), int(x1), int(y1)), "depth": depth, "parent": parent, "children": []})
    if parent >= 0:
        out[parent]["children"].append(me)
    if depth >= 4:
        return
    sub = ink[y0:y1, x0:x1]
    best = None
    for axis in (0, 1):  # 0: split rows (horizontal gaps), 1: split columns
        prof = sub.sum(axis=1) if axis == 0 else sub.sum(axis=0)
        noise = max(1, int(0.004 * (sub.shape[1] if axis == 0 else sub.shape[0])))
        empty = prof <= noise
        # runs of empty rows/cols, interior only
        runs, i, n = [], 0, len(empty)
        while i < n:
            if empty[i]:
                j = i
                while j < n and empty[j]:
                    j += 1
                if i > 0 and j < n and j - i >= min_gap:
                    runs.append((i, j))
                i = j
            else:
                i += 1
        if runs:
            width = max(b - a for a, b in runs)
            if best is None or width > best[0]:
                best = (width, axis, runs)
    if best is None:
        return
    _, axis, runs = best
    edges = [0] + [c for r in runs for c in r] + [sub.shape[0] if axis == 0 else sub.shape[1]]
    for a, b in zip(edges[::2], edges[1::2]):
        child = (x0, y0 + a, x1, y0 + b) if axis == 0 else (x0 + a, y0, x0 + b, y1)
        _xy_cut(ink, child, min_gap, out, depth + 1, me)


def find_views(rgb, work=1200, max_views=8):
    """Candidate drawing regions on the sheet: the XY-cut tree of the ink (frame and title-block rules removed).
    Returns dicts {box (x0, y0, x1, y1) in rgb px, depth, parent, children} for nodes big enough to be a drawing,
    biggest first. The root (= all drawings on the sheet, cropped to their ink) is always included."""
    h, w = rgb.shape[:2]
    s = work / max(h, w)
    g = cv2.cvtColor(cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    full = ink_mask(g)
    gh, gw = full.shape
    m = int(0.02 * max(gh, gw))  # scanner edges, colour charts and rulers live in the margin
    full[:m] = 0; full[-m:] = 0; full[:, :m] = 0; full[:, -m:] = 0
    full = cv2.morphologyEx(full, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # dust
    ink = remove_rules(full)
    nodes = []
    _xy_cut(ink, (0, 0, gw, gh), max(6, int(0.009 * max(gh, gw))), nodes)
    if not nodes:  # blank sheet
        return [{"box": (0, 0, w, h), "depth": 0, "parent": -1, "children": []}]
    keep = []
    for i, nd in enumerate(nodes):
        x0, y0, x1, y1 = nd["box"]
        bw, bh = x1 - x0, y1 - y0
        big = bw * bh >= 0.02 * gh * gw and min(bw, bh) >= 0.06 * max(gh, gw)
        if i == 0 or big:
            keep.append(i)
    keep = sorted(keep, key=lambda i: -_area(nodes[i]["box"]))[:max_views]
    if 0 not in keep:
        keep = [0] + keep[:-1]
    idx = {old: new for new, old in enumerate(keep)}
    pad = int(0.01 * max(gh, gw))
    grow = int(0.03 * max(gh, gw))
    _, comp = cv2.connectedComponents(full, connectivity=8)
    views = []
    for old in keep:
        nd = nodes[old]
        x0, y0, x1, y1 = nd["box"]
        # the rule remover also strips a plan's long outer walls: grow the box over the full ink connected to it
        wx0, wy0, wx1, wy1 = max(0, x0 - grow), max(0, y0 - grow), min(gw, x1 + grow), min(gh, y1 + grow)
        inner = np.unique(comp[y0:y1, x0:x1])
        win = np.isin(comp[wy0:wy1, wx0:wx1], inner[inner > 0])
        if win.any():
            ys, xs = np.nonzero(win)
            x0, y0, x1, y1 = min(x0, wx0 + xs.min()), min(y0, wy0 + ys.min()), max(x1, wx0 + xs.max() + 1), max(y1, wy0 + ys.max() + 1)
        # nearest kept ancestor, kept descendants
        p = nd["parent"]
        while p >= 0 and p not in idx:
            p = nodes[p]["parent"]
        views.append({"box": (max(0, int((x0 - pad) / s)), max(0, int((y0 - pad) / s)), min(w, int((x1 + pad) / s)), min(h, int((y1 + pad) / s))),
                      "depth": nd["depth"], "parent": idx.get(p, -1)})
    for i, v in enumerate(views):
        v["children"] = [j for j, u in enumerate(views) if u["parent"] == i]
    return views


def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _aspect(b):
    """Long side / short side of a box (>= 1)."""
    w, h = max(1, b[2] - b[0]), max(1, b[3] - b[1])
    return max(w / h, h / w)


# ---------------------------------------------------------------- network

_device = None
_model = None


def _net():
    global _device
    import torch
    import cubicasa_model as M
    global _model
    if _model is None:  # own copy: cubicasa_model's shared instance stays on the CPU for the CubiCasa path
        import copy
        _model = copy.deepcopy(M.load_model())
    model = _model
    if _device is None:
        _device = "cpu"
        if torch.cuda.is_available() and os.environ.get("STOA_ARCHIVE_DEVICE", "cuda") == "cuda":
            try:
                model.to("cuda")
                _device = "cuda"
            except RuntimeError:
                model.to("cpu")
    return model


def predict(rgb):
    """Network outputs for an RGB crop at its own resolution: heatmaps (21,h,w), rooms (12,h,w), icons (11,h,w) softmax."""
    global _device
    import torch
    import torch.nn.functional as F
    import cubicasa_model as M
    model = _net()
    h, w = rgb.shape[:2]
    x = torch.from_numpy(np.ascontiguousarray(np.moveaxis(rgb, -1, 0))).float().unsqueeze(0)
    x = 2 * (x / 255.0) - 1
    with torch.no_grad():
        try:
            out = model(x.to(_device)).float().cpu()
        except RuntimeError as e:  # out of GPU memory (the GPU is shared): fall back to CPU for good
            if _device != "cuda":
                raise
            torch.cuda.empty_cache()
            model.to("cpu"); _device = "cpu"
            out = model(x)
    pp = M._load_module("cc_post", "floortrans/post_prosessing.py")
    return pp.split_prediction(out, (h, w), M.SPLIT)


# ---------------------------------------------------------------- features

def features(rooms, icons, gray):
    """Plan-likeness signals from the network's own outputs (rooms/icons softmax) and the crop's ink."""
    am = rooms.argmax(0)
    n = am.size
    wall = am == WALL
    inter = np.isin(am, INTERIOR)
    f = {"wall_frac": float(wall.mean()), "interior_frac": float(inter.mean())}
    k = np.ones((3, 3), np.uint8)
    wall8 = wall.astype(np.uint8)
    near_wall = cv2.dilate(wall8, k, iterations=3)
    # rooms the predicted walls actually enclose: interior blobs whose rim is mostly wall
    nc, lab, st, _ = cv2.connectedComponentsWithStats(inter.astype(np.uint8), 4)
    min_a = 0.002 * n
    span = 0
    if wall.any():
        wx = np.nonzero(wall.any(0))[0]
        span = wx.max() - wx.min() + 1
    closed = comps = closed_area = band = 0
    for i in range(1, nc):
        if st[i][4] < min_a:
            continue
        comps += 1
        x, y, bw, bh, a = st[i]
        m = (lab[y:y + bh, x:x + bw] == i).astype(np.uint8)
        m = cv2.copyMakeBorder(m, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0)
        nw = cv2.copyMakeBorder(near_wall[y:y + bh, x:x + bw], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0)
        ring = cv2.dilate(m, k, iterations=2) - m
        if ring.sum() and nw[ring > 0].mean() > 0.6:
            closed += 1
            closed_area += a
            if span and bw >= 0.7 * span:  # elevations: the "rooms" are storeys as wide as the building
                band += a
    f.update(rooms_big=comps, rooms_closed=closed, closed_frac=float(closed_area / n),
             band_frac=float(band / closed_area) if closed_area else 0.0)
    # plan rooms are mostly blank paper; elevation/section "rooms" are full of windows, siding, shading, beams
    ink = ink_mask(gray)
    core = cv2.erode(inter.astype(np.uint8), k, iterations=3) > 0
    f["room_ink"] = float(ink[core].mean()) if core.any() else 1.0
    f["ink_frac"] = float(ink.mean())
    if wall.any():
        dt = cv2.distanceTransform(wall8, cv2.DIST_L2, 3)
        ridge = (dt >= cv2.dilate(dt, k)) & (dt > 0)
        t = 2 * dt[ridge]
        f["wall_px"] = float(np.median(t)) if t.size else 0.0
    else:
        f["wall_px"] = 0.0
    ia = icons.argmax(0)
    f["door_frac"] = float((ia == 2).mean())
    f["window_frac"] = float((ia == 1).mean())
    return f


# ---------------------------------------------------------------- printed view titles (OCR, no LLM)
# Archive sheets usually print what each drawing is ("SECTION B-B", "LEIKKAUS / SKARNING", "Grundriss ...").
# RapidOCR (ONNX, CPU) reads them; matching is plain keyword stems in the dataset's languages.
NOT_PLAN_STEMS = ("section", "elevation", "facade", "fasad", "julkisivu", "leikkaus", "karning", "snitt", "schnitt",
                  "ansicht", "aufriss", "coupe", "prospetto", "sezione", "alzato", "alzado", "seccion", "fachada",
                  "doorsnede", "doorsnee", "gevel", "aanzicht", "perspectiv", "perspektiv", "isometric", "axonometric",
                  "detail", "profile", "elevacion")
SITE_STEMS = ("siteplan", "plotplan", "location", "vicinity", "lageplan", "situationsplan", "asemapiirros",
              "asemakaava", "kartta", "map", "plandesituation")
PLAN_STEMS = ("floorplan", "groundfloor", "firstfloor", "secondfloor", "basement", "grundriss", "erdgeschoss",
              "obergeschoss", "pohja", "kerros", "plattegrond", "pianta", "planta", "rezdechaussee", "etage", "vaning",
              "bottenvaning", "kellari", "keller", "kallare", "cellar", "planof", "plandu", "plande", "plan")
LEVEL_RE = re.compile(r"^[+\-]\s*\d{0,3}[.,]\d{2,3}$")

_ocr = None


def _norm(t):
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9+\-., ]", " ", t).strip()


def sheet_text(rgb, work=2000):
    """Text lines on the sheet: [{'box': (x0, y0, x1, y1) in rgb px, 'text': str}], [] if OCR is unavailable."""
    global _ocr
    try:
        if _ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            _ocr = RapidOCR()
    except Exception:
        return []
    s = min(1.0, work / max(rgb.shape[:2]))
    im = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else rgb
    res, _ = _ocr(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    out = []
    for quad, text, conf in res or []:
        if float(conf) < 0.6:
            continue
        xs, ys = [q[0] / s for q in quad], [q[1] / s for q in quad]
        out.append({"box": (min(xs), min(ys), max(xs), max(ys)), "text": text})
    return out


def attach_labels(views, texts):
    """Give each text line to the smallest view whose box (stretched down for a title underneath) holds its centre."""
    for v in views:
        v["labels"] = []
    for t in texts:
        cx, cy = (t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2
        best = None
        for i, v in enumerate(views):
            x0, y0, x1, y1 = v["box"]
            h = y1 - y0
            if x0 <= cx <= x1 and y0 - 0.06 * h <= cy <= y1 + 0.15 * h:
                if best is None or _area(v["box"]) < _area(views[best]["box"]):
                    best = i
        if best is not None:
            views[best]["labels"].append(t["text"])


def read_labels(views, i):
    """What the printed titles in view i (and inside it) say: ('not_plan' | 'site' | 'levels' | 'plan' | None, evidence)."""
    lines = list(views[i].get("labels", []))
    for j in _descendants(views, i):
        lines += views[j].get("labels", [])
    found = {"not_plan": [], "site": [], "plan": [], "levels": []}
    for line in lines:
        n = _norm(line)
        if LEVEL_RE.match(n.replace(" ", "")):
            found["levels"].append(line)
            continue
        words = n.split()
        if not words or len(words) > 5:  # titles are short; long lines are notes ("see section A on sheet 4")
            continue
        flat = "".join(w for w in words if w.isalpha())
        if any(st in flat for st in SITE_STEMS):
            found["site"].append(line)
        elif "elev" in words or any(w.startswith(st) or (len(st) > 5 and st in w) for w in words for st in NOT_PLAN_STEMS):
            found["not_plan"].append(line)
        elif any(st in flat for st in PLAN_STEMS):
            found["plan"].append(line)
    if found["not_plan"] and not found["plan"]:
        return "not_plan", found["not_plan"][0]
    if found["site"] and not found["plan"]:
        return "site", found["site"][0]
    if len(found["levels"]) >= 3 and not found["plan"]:
        return "levels", ", ".join(found["levels"][:3])
    if found["plan"] and not found["not_plan"]:
        return "plan", found["plan"][0]
    return None, None


# ---------------------------------------------------------------- decision
# Thresholds fit by hand on the seed-7 sample (60 sheets) and checked on the seed-11 sample (60 sheets):
# _analysis/archive_eval.py. Precision first: a wrong 3D model is worse than "no 3D".
MIN_ROOMS, MIN_CLOSED, MAX_ROOM_INK, MAX_BAND, MAX_ASPECT = 3, 0.08, 0.06, 0.5, 2.4


def verdict(f, label=(None, None), aspect=1.0):
    """'plan' | 'other' (evidence of another drawing type: elevation, section, site plan) | 'unclear' (too little
    structure: a fragment, text, ornament, photo) + why. Only 'other' may veto a plan region that contains it."""
    kind, said = label
    if kind == "not_plan":
        return "other", f'is titled "{said}"'
    if kind == "site":
        return "other", f'is titled "{said}", a site or location plan'
    if kind == "levels":
        return "other", f"carries height levels ({said}), like a section or elevation"
    rooms, closed, ink = MIN_ROOMS, MIN_CLOSED, MAX_ROOM_INK
    if kind == "plan":  # a printed "floor plan" title lowers the bar a little
        rooms, closed, ink = 2, 0.05, 0.08
    if f["rooms_closed"] >= rooms and f["closed_frac"] >= closed:
        if max(aspect, 1 / aspect) > MAX_ASPECT:
            return "unclear", "is a long strip (a row of small drawings or a cut-off part)"
        if f["band_frac"] > MAX_BAND and kind != "plan":
            return "other", "has enclosed areas that are full-width bands, like the storeys of an elevation"
        if f["room_ink"] > ink:
            return "other", "has enclosed areas full of linework (windows, siding, hatching), like an elevation or section"
        return "plan", (f'has rooms enclosed by walls and is titled "{said}"' if said else "has rooms enclosed by walls")
    if f["rooms_closed"] >= 2 or f["wall_frac"] >= 0.04:
        return "unclear", "has walls that do not enclose several rooms"
    return "unclear", "has no walls enclosing rooms"


def score(f):
    return f["closed_frac"] * min(f["rooms_closed"], 12) * (1 - min(f["room_ink"], 0.1) * 5)


def net_size(box):
    """Network input long side for a view: the view's own pixels, kept inside what the network handles well."""
    L = max(box[2] - box[0], box[3] - box[1])
    return int(min(1400, max(640, L)))


def _descendants(views, i):
    out = []
    for j in views[i]["children"]:
        out += [j] + _descendants(views, j)
    return out


def decide(views):
    """Sheet-level choice from per-view verdicts. Returns (best view index or None, number of usable plans).
    A view is usable if it looks like a plan and no big part of it is another drawing (elevation, section);
    a usable view with a usable descendant gives way to it (one plan per 3D model)."""
    area = lambda i: _area(views[i]["box"])
    usable = []
    for i, v in enumerate(views):
        if v["verdict"] != "plan":
            continue
        v["blocked_by"] = [j for j in _descendants(views, i) if views[j]["verdict"] == "other" and area(j) >= 0.15 * area(i)]
        # several separate drawings side by side that are not plans themselves: a multi-view sheet whose
        # plan could not be isolated (fragments of one plan split into strips, so they do not count)
        v["siblings"] = [j for j in v["children"] if views[j]["verdict"] != "plan" and area(j) >= 0.1 * area(i)
                         and _aspect(views[j]["box"]) <= 3.0]
        if not v["blocked_by"] and len(v["siblings"]) < 2:
            usable.append(i)
    leaves = [i for i in usable if not any(j in usable for j in _descendants(views, i))]
    best = max(leaves, key=lambda i: views[i]["score"]) if leaves else None
    return best, len(leaves)


def judge(views):
    """(Re)compute verdicts and scores from cached features and labels, then decide."""
    for i, v in enumerate(views):
        x0, y0, x1, y1 = v["box"]
        v["label"] = read_labels(views, i)
        v["verdict"], v["why"] = verdict(v["features"], v["label"], (x1 - x0) / max(1, y1 - y0))
        v["score"] = score(v["features"]) if v["verdict"] == "plan" else 0.0
    return decide(views)


def analyse_sheet(path, keep_pred=True):
    """Find the floor plan on an archive sheet. Returns a dict with every candidate view (box, features, verdict),
    'best' (index or None), 'plans' (usable plan views), 'region' (best box in original-image pixels),
    'pred' = (heatmaps, rooms, icons, net_rgb) of the best view, or 'reason' when there is no plan."""
    with Image.open(path) as im:
        orig_wh = im.size
    rgb = load_rgb(path)
    views = find_views(rgb)
    attach_labels(views, sheet_text(rgb))
    preds = {}
    for i, v in enumerate(views):
        x0, y0, x1, y1 = v["box"]
        crop = rgb[y0:y1, x0:x1]
        s = net_size(v["box"]) / max(crop.shape[:2])
        c = cv2.resize(crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))),
                       interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        heat, rooms, icons = predict(c)
        v["features"] = features(rooms, icons, cv2.cvtColor(c, cv2.COLOR_RGB2GRAY))
        v["net_scale"] = s  # network px per loaded-rgb px
        if keep_pred:
            preds[i] = (heat, rooms, icons, c)
    best, nplans = judge(views)
    res = {"path": path, "orig_wh": orig_wh, "rgb_hw": rgb.shape[:2], "views": views, "best": best, "plans": nplans}
    if best is None:
        res["reason"] = _reason(views)
    else:
        res["pred"] = preds.get(best)
        x0, y0, x1, y1 = views[best]["box"]
        k = orig_wh[0] / rgb.shape[1]
        res["region"] = [round(x0 * k), round(y0 * k), round(x1 * k), round(y1 * k)]  # original-image pixels
    return res


def _reason(views):
    """One sentence for the app when there is no usable plan, from the most telling view."""
    if any(v["verdict"] == "plan" for v in views):
        return ("No usable floor plan: the part that looks like a plan also holds other drawings "
                "(elevations or sections) that could not be separated from it.")
    others = [v for v in views if v["verdict"] == "other"]
    if others:
        d = max(others, key=lambda v: _area(v["box"]))
        return f"Not a floor plan: the drawing {d['why']}."
    structured = [v for v in views if v["features"]["wall_frac"] >= 0.04 or v["features"]["rooms_closed"] >= 2]
    if structured:
        return ("No floor plan found: the drawing has wall-like lines, but they do not enclose rooms. "
                "It looks like an elevation, section, detail or site map.")
    return "No floor plan found: nothing on the sheet has walls enclosing rooms (text, photo, map or ornament?)."
