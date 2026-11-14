"""3D scene for a raster plan whose counts come from the YOLO detectors (an uploaded plan, or a CubiCasa plan).

Walls, rooms and wall openings come from stoa_qa.floorplan (the CubiCasa5K network's wall map on a rescaled image,
plus thin drawn partitions). Every item the answers count is placed from the same detector box the answer marks on the
sheet, under the same id "yolo:<index into the plan's boxes>": doors and windows are set into the wall they sit in
(cut through it, leaf opening towards the drawn swing), fixtures stand where detected, and a counted room box
(kitchen, bedroom, WC, bathroom, balcony) takes over the traced room it matches, or is cut out of it. So "How many
doors?" lights up the same doors on the sheet and in 3D. On a new drawing a door or window found in no wall is
misread furniture: it is neither built nor counted (stoa_qa.cubicasa), so the answer and the model agree.

One reconstruction per image and detector openings, cached in eval_out/plan3d/ and shared by counting and 3D.
Approximate, like every raster scene: scale from the median door opening (0.9 m), storey heights assumed.
Needs the raster extra, third_party/CubiCasa5k and third_party/weights/model_best_val_loss_var.pkl (SETUP.md step 4).
"""
from __future__ import annotations

import importlib.util
import statistics
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import cubicasa
from .cubicasa_rules import CLASSES, QUESTION_CLASSES

REPO = Path(__file__).resolve().parent.parent
NET_REPO = REPO / "third_party" / "CubiCasa5k"
NET_WEIGHTS = REPO / "third_party" / "weights" / "model_best_val_loss_var.pkl"
ENGINE = "cubicasa5k-model+yolo"
HEIGHTS = {"storey_height": 2.6, "door_head": 2.1, "window_sill": 0.9, "window_head": 2.1, "fixture_height": 0.9}
DOOR_M, WALL_M = 0.9, 0.2
NET_ROOM_KIND = {"Kitchen": "kitchen", "Living Room": "living", "Bed Room": "bedroom", "Bath": "bath", "Entry": "corridor",
                 "Storage": "storage", "Outdoor": "outdoor", "Garage": "other", "Undefined": "other"}
# detector class -> (scene group, room kind or fixture label)
YOLO_ROOM = {"kitchen": ("kitchen", "Kitchen"), "bedroom": ("bedroom", "Bedroom"), "wc_room": ("bath", "WC"),
             "bathroom": ("bath", "Bathroom"), "balcony": ("outdoor", "Balcony")}
YOLO_FIXTURE = {"toilet": "Toilet", "sink": "Sink", "shower": "Shower", "bathtub": "Bathtub", "sauna": "Sauna"}
COVER = 0.5  # a room box takes over the traced room when each covers at least this share of the other


@lru_cache(maxsize=1)
def available() -> bool:
    """The CubiCasa5K network can run here (torch, OpenCV, its code and weights)."""
    return all(importlib.util.find_spec(m) for m in ("torch", "cv2", "shapely")) and NET_REPO.is_dir() and NET_WEIGHTS.is_file()


def box_id(k: int) -> str:
    return f"yolo:{k}"


def counted(boxes: list, cfg: dict) -> dict[int, str]:
    """Index into `boxes` -> detector class, for every box some question counts (the val-tuned threshold and cut)."""
    pos = {id(b): k for k, b in enumerate(boxes)}
    out: dict[int, str] = {}
    for q in QUESTION_CLASSES:
        for b in cubicasa._yolo_selected(boxes, q, cfg):
            out[pos[id(b)]] = CLASSES[b[0]]
    return out


VERSION = "fp3"  # bump when stoa_qa.floorplan changes what it reconstructs
PLANS = REPO / "eval_out" / "plan3d"


def openings_of(boxes: list, cfg: dict) -> list:
    """The counted door / window boxes as floorplan.reconstruct takes them: (index, kind, x0, y0, x1, y1)."""
    return [(k, c, *boxes[k][2:6]) for k, c in counted(boxes, cfg).items() if c in ("door", "window")]


def plan_for(image: Path, openings: list | None):
    """floorplan.reconstruct for one image and its detector openings, cached on disk (counting and 3D share it)."""
    import hashlib
    import json

    from . import cubicasa_live
    from . import floorplan as FP

    sig = hashlib.sha1(json.dumps(openings, default=float).encode()).hexdigest()[:10]
    f = PLANS / f"{cubicasa_live.image_hash(image)}-{sig}-{VERSION}.json"
    if f.exists():
        return FP.from_json(json.loads(f.read_text()))
    plan = FP.reconstruct(FP.read_rgb(image), openings)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(FP.to_json(plan)))
    return plan


def off_wall(image: Path, boxes: list, cfg: dict) -> set[int]:
    """Indices of counted door / window boxes that sit in no wall (furniture the detector misread)."""
    plan = plan_for(image, openings_of(boxes, cfg))
    return {o.index for o in plan.openings if not o.on_wall and o.index is not None}


def _rect(x0, y0, x1, y1) -> list:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _scale_plan(plan) -> tuple[float, str]:
    """Pixels per metre: the median width of the doors found in walls is 0.9 m; else walls are 0.2 m thick."""
    widths = [o.width for o in plan.openings if o.kind == "door" and o.on_wall and o.width > 0]
    if len(widths) >= 2:
        return statistics.median(widths) / DOOR_M, "median door opening = 0.9 m"
    if plan.thick > 0:
        return plan.thick / WALL_M, "wall thickness = 0.2 m"
    return 50.0, "default 50 px/m"


def _leaf(rect: list, horizontal: bool, side: int, t: float) -> list:
    """An open door leaf (px): hinged at one end of the opening, standing across it on the swing side."""
    xs, ys = [p[0] for p in rect], [p[1] for p in rect]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if horizontal:
        w, yc = x1 - x0, (y0 + y1) / 2
        ya, yb = sorted((yc, yc + side * w))
        return _rect(x0, ya, x0 + t, yb)
    h, xc = y1 - y0, (x0 + x1) / 2
    xa, xb = sorted((xc, xc + side * h))
    return _rect(xa, y0, xb, y0 + t)


def build(image: Path, rel: str, boxes: list | None, cfg: dict, plan=None, openings: list | None = None) -> dict:
    """Scene dict (blueprint3d/scene@1) for one plan image. `boxes` None: the network's own doors and windows.
    `openings`: the door / window boxes the reconstruction was made with (the counting step's, so both share it)."""
    from shapely.geometry import Polygon
    from shapely.geometry import box as sbox

    from . import floorplan as FP

    rgb = FP.read_rgb(image)
    h = rgb.shape[0]
    sel = counted(boxes, cfg) if boxes else {}
    if openings is None and boxes:
        openings = openings_of(boxes, cfg)
    plan = plan if plan is not None else plan_for(Path(image), openings)
    ppm, scale_from = _scale_plan(plan)
    P = lambda pts: [[round(float(x) / ppm, 3), round((h - float(y)) / ppm, 3)] for x, y in pts]  # px -> metres, y up

    walls = [{"id": f"wall:{i}", "layer": "PLAN_WALL", "poly": P(o), "holes": [P(hh) for hh in hs]}
             for i, (o, hs) in enumerate(plan.walls)]
    rooms = []  # "_geom": the room in image pixels (shapely), turned into poly / holes at the end
    for r in plan.rooms:
        kind = NET_ROOM_KIND.get(r["cls"], "other")
        g = Polygon(r["poly"], r["holes"]).buffer(0)
        if not g.is_empty:
            rooms.append({"id": f"room:{len(rooms) + 1}", "layer": "PLAN_ROOM", "number": "", "label": r["cls"],
                          "kind": kind, "outdoor": kind == "outdoor", "printed_area": None, "unit": None, "_geom": g, "_net": True})
    doors, windows, fixtures = [], [], []
    leaf_t = max(2.0, 0.05 * ppm)
    for i, op in enumerate(plan.openings):
        if not op.on_wall:  # a "door" in the middle of a room is furniture the detector misread: no opening for it
            continue
        oid = box_id(op.index) if op.index is not None else f"{op.kind}:{i}"
        if op.kind == "door":
            side = 1
            if op.index is not None and boxes:  # the detector box holds the swing arc: the leaf opens to that side
                b = boxes[op.index]
                c = np.mean(op.rect, axis=0)
                side = 1 if ((b[3] + b[5]) / 2 >= c[1] if op.horizontal else (b[2] + b[4]) / 2 >= c[0]) else -1
            doors.append({"id": oid, "layer": "PLAN_DOOR", "leaf": P(_leaf(op.rect, op.horizontal, side, leaf_t)),
                          "lintel": P(op.rect), "width_m": round(op.width / ppm, 2)})
        else:
            windows.append({"id": oid, "layer": "PLAN_WINDOW", "poly": P(op.rect)})
    if boxes:
        claimed: set[int] = set()
        for k in sorted(sel, key=lambda k: -boxes[k][1]):
            cls, (_, _, x0, y0, x1, y1) = sel[k], boxes[k]
            if cls in YOLO_FIXTURE:
                fixtures.append({"id": box_id(k), "layer": "YOLO_FIXTURE", "label": YOLO_FIXTURE[cls],
                                 "poly": P(_rect(x0, y0, x1, y1)), "score": boxes[k][1]})
            elif cls in YOLO_ROOM:
                kind, label = YOLO_ROOM[cls]
                b = sbox(x0, y0, x1, y1)

                def mutual(r):  # how well box and traced room cover each other (the smaller of the two shares)
                    inter = r["_geom"].intersection(b).area
                    return min(inter / max(b.area, 1e-9), inter / max(r["_geom"].area, 1e-9))
                free = [i for i, r in enumerate(rooms) if r.get("_net") and i not in claimed and not r["_geom"].is_empty]
                best = max(free, key=lambda i: mutual(rooms[i]), default=None)
                if best is not None and mutual(rooms[best]) >= COVER:  # the detected room is this traced room: one element, one id
                    claimed.add(best)
                    rooms[best].update(id=box_id(k), layer="YOLO_ROOM", label=label, kind=kind, outdoor=kind == "outdoor",
                                       score=boxes[k][1])
                else:  # e.g. an open kitchen inside a traced living room: cut it out, so the floors never overlap
                    for i in (i for i, r in enumerate(rooms) if r.get("_net")):
                        if rooms[i]["_geom"].intersects(b):
                            rooms[i]["_geom"] = rooms[i]["_geom"].difference(b)
                    rooms.append({"id": box_id(k), "layer": "YOLO_ROOM", "number": "", "label": label, "kind": kind,
                                  "outdoor": kind == "outdoor", "printed_area": None, "unit": None, "_geom": b,
                                  "score": boxes[k][1]})
    kept = []
    for r in rooms:
        g = r.pop("_geom")
        r.pop("_net", None)
        if g.geom_type == "MultiPolygon":  # a traced room split by a cut-out keeps its largest part
            g = max(g.geoms, key=lambda p: p.area)
        if g.is_empty or g.geom_type != "Polygon" or g.area < 1.0:
            continue
        at = g.representative_point()
        r.update(poly=P(list(g.exterior.coords)[:-1]), holes=[P(list(hh.coords)[:-1]) for hh in g.interiors],
                 area_m2=round(g.area / ppm ** 2, 2), label_at=P([[at.x, at.y]])[0])
        kept.append(r)
    rooms = kept
    for n, r in enumerate(rooms, 1):
        r["number"] = str(n)
    pts = [q for g in (walls, rooms) for el in g for q in el["poly"]] or [[0, 0], [1, 1]]
    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
    how = ("walls and rooms traced from the CubiCasa5K network's wall map; doors and windows set into those walls and "
           "fixtures and room types placed from the YOLO detector boxes the answers count" if boxes
           else "everything from the CubiCasa5K network")
    return {
        "schema": "blueprint3d/scene@1",
        "source": {"file": rel, "format": "raster", "engine": ENGINE if boxes else "cubicasa5k-model", "units": "m",
                   "pixels_per_m": round(ppm, 2), "scale_from": scale_from, "rescaled": round(plan.factor, 2)},
        "assumptions": {**HEIGHTS, "note": f"Approximate: {how}; scale from {scale_from}; heights assumed."},
        "bbox": [min(xs), min(ys), max(xs), max(ys)],
        "counts": {"rooms": len(rooms), "walls": len(walls), "doors": len(doors), "windows": len(windows), "units": 0,
                   "fixtures": len(fixtures)},
        "units": [], "rooms": rooms, "walls": walls, "doors": doors, "windows": windows, "fixtures": fixtures,
    }


def for_drawing(ask_path: str, model: str) -> dict:
    """The scene of a raster drawing, with the detector boxes its answers use when there are any."""
    image = cubicasa.image_of(model)
    if image is None:
        raise FileNotFoundError(model)
    yolo = cubicasa.predictions_for(ask_path).get("yolo")
    # the thresholds of whichever detector made the boxes (uploads: cubicasa_live_picks.json), as the answers use
    return build(image, model, yolo["boxes"] if yolo else None, (yolo or {}).get("cfg") or cubicasa.picks(),
                 openings=(yolo.get("live") or {}).get("openings") if yolo else None)
