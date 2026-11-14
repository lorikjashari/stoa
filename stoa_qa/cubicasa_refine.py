"""Raster counts that the polygon assembly gets wrong.

The CubiCasa network's class maps are more reliable than the rectangles it builds
for doors, windows, sinks and toilets: the assembler drops openings and almost
never emits a bathtub. Showers are not a class at all, and a wet room on these
plans holds one shower or tub. Two storeys drawn on one sheet are two groups of
rooms with a gap between them.

Thresholds and the floor gap were chosen on CubiCasa's val split. The held-out
test split is not used to set them.
"""
from __future__ import annotations

# Min component area at the network's working resolution (longest side <= 1024).
# Each value is the peak of a sweep on the val split; neighbours scored within a point of it.
ICON_MIN_AREA = {"Door": 40, "Window": 100, "Sink": 80}
BED_MIN_AREA = 7000
# question -> icon class. Only questions where components beat assembled polygons on val.
SEG_QUESTIONS = {
    "doors": "Door",
    "windows": "Window",
    "sinks": "Sink",
}
FLOOR_GAP = 0.04
# a wet room smaller than this fraction of the largest wet room is a separate WC.
# Chosen on the val split: it beat a fixed area cut, and a lone wet room stays a bathroom.
WC_OF_LARGEST = 0.5


def component_count(areas: list[int], minimum: int) -> int:
    return sum(a >= minimum for a in areas)


def _room_boxes(rooms: list[dict]) -> list[tuple[float, float, float, float]]:
    boxes = []
    for r in rooms:
        if r.get("class") == "Outdoor":
            continue
        pts = r.get("poly") or []
        if len(pts) < 3:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes


def _groups(boxes: list[tuple[float, float, float, float]], axis: int, gap_frac: float) -> list[list[float]]:
    """Spans along one axis, with only the gaps large enough to separate two storeys kept."""
    spans = sorted((b[axis], b[axis + 2]) for b in boxes)
    merged = [list(spans[0])]
    for a, c in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], c)
        else:
            merged.append([a, c])
    width = merged[-1][1] - merged[0][0]
    if width <= 0:
        return [merged[0]]
    kept = [merged[0]]
    for i in range(len(merged) - 1):
        if merged[i + 1][0] - merged[i][1] > gap_frac * width:
            kept.append(merged[i + 1])
        else:
            kept[-1][1] = max(kept[-1][1], merged[i + 1][1])
    return kept


def floor_plates(rooms: list[dict], gap_frac: float = FLOOR_GAP) -> list[list[list[float]]]:
    """One box per storey drawn on the sheet, in the same pixel space as the room polygons."""
    boxes = _room_boxes(rooms)
    if not boxes:
        return []
    horizontal, vertical = _groups(boxes, 0, gap_frac), _groups(boxes, 1, gap_frac)
    if len(vertical) > len(horizontal):
        groups, x0, x1 = vertical, min(b[0] for b in boxes), max(b[2] for b in boxes)
        return [[[round(x0, 1), round(a, 1)], [round(x1, 1), round(c, 1)]] for a, c in groups]
    groups, y0, y1 = horizontal, min(b[1] for b in boxes), max(b[3] for b in boxes)
    return [[[round(a, 1), round(y0, 1)], [round(c, 1), round(y1, 1)]] for a, c in groups]


def floor_count(rooms: list[dict], gap_frac: float = FLOOR_GAP) -> int:
    """How many storeys are drawn on the sheet: groups of rooms separated by a real gap."""
    boxes = _room_boxes(rooms)
    if len(boxes) < 2:
        return 1
    return max(len(_groups(boxes, 0, gap_frac)), len(_groups(boxes, 1, gap_frac)))


def _area(pts: list) -> float:
    if len(pts) < 3:
        return 0.0
    acc = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        acc += x0 * y1 - x1 * y0
    return abs(acc) / 2


def wet_kinds(rooms: list[dict], scale: float = 1.0) -> list[tuple[str, dict]]:
    """('wc' | 'bathroom', room) for every room the network called Bath.

    A wet room clearly smaller than the largest one is a separate WC. Each wet room
    is one shower or bathtub: the network has no shower class.
    """
    wet = [(_area(r["poly"]) * scale * scale, r) for r in rooms if r.get("class") == "Bath" and r.get("poly")]
    if not wet:
        return []
    largest = max(a for a, _ in wet)
    return [("wc" if a < WC_OF_LARGEST * largest else "bathroom", r) for a, r in wet]


def wet_rooms(rooms: list[dict], scale: float = 1.0) -> tuple[int, int, int]:
    """(showers_and_tubs, wc_rooms, bathrooms)."""
    kinds = wet_kinds(rooms, scale)
    wc = sum(k == "wc" for k, _ in kinds)
    return len(kinds), wc, len(kinds) - wc


def seg_counts(icons: dict[str, list[int]], rooms: dict[str, list[int]] | None = None) -> dict[str, int]:
    """Question -> count, from component area lists (working-resolution pixels)."""
    out = {key: component_count(icons.get(cls, []), ICON_MIN_AREA[cls]) for key, cls in SEG_QUESTIONS.items()}
    if rooms is not None:
        out["bedrooms"] = component_count(rooms.get("Bed Room", []), BED_MIN_AREA)
    return out


def boxes_from_mask(mask, minimum: int, scale: float) -> list[list[list[float]]]:
    """Connected components -> boxes in original image pixels, as two corners."""
    import cv2

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"), 8)
    boxes = []
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < minimum:
            continue
        x, y, w, h = (int(stats[i, j]) for j in range(4))
        boxes.append([[round(x / scale, 1), round(y / scale, 1)], [round((x + w) / scale, 1), round((y + h) / scale, 1)]])
    return boxes


def apply(pred: dict, seg: dict | None = None) -> dict[str, int]:
    """Counts for one plan. Segmentation counts replace polygon counts where that won on val."""
    counts = dict(pred.get("counts") or {})
    rooms = pred.get("rooms") or []
    scale = float(pred.get("scale") or 1.0)
    fixtures, wc, bathrooms = wet_rooms(rooms, scale)
    counts["baths"] = fixtures
    counts["wc_rooms"] = wc
    counts["bathrooms"] = bathrooms
    counts["floors"] = floor_count(rooms)
    if seg is not None:
        counts.update(seg_counts(seg.get("icons") or {}, seg.get("rooms")))
    return counts
