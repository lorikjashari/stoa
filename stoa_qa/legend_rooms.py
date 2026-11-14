"""Rooms counted by their legend colour on a colour-coded raster sheet (no LLM, no detector).

Many design sheets fill each room with the colour of its type and print a legend ("DORMITORY" next to a light-green
swatch). Then the drawn rooms of a type are the patches of that colour on the plan: the walls (drawn with a white or
dark band) separate neighbouring rooms, and the fill flows around furniture, so each room is one patch. This counts
the rooms actually drawn, not a number printed in a table, and every counted room is outlined on the sheet.

  1. legend: an OCR line (stoa_qa.sheet_ocr) with a solid, non-grey colour swatch just left of it
  2. which entries answer the question: room-type words (stoa_qa.room_labels + dormitory / guest room words)
  3. drawings: the legend-coloured patches grouped by proximity; insets with far less colour than the main plan
     (a KEY PLAN repeating the wing colours) are left out
  4. rooms: compact patches of that colour (within TOL per channel), away from the legend itself, at least
     MIN_SHARE of the largest patch (drops offcuts in closets and door swings) and at least MIN_SIDE px each way
     (drops colour fringes along lines, e.g. a sheet border)

Checked on the USAF dormitory sheet A-2: 46 'DORMITORY' rooms (4,013-4,183 px each) + 2 'ACCESSIBLE (ABA) DORMITORY'
rooms (~8,000 px) = 48, the number the sheet's own room-count table gives for that floor.
"""
from __future__ import annotations

import re
from pathlib import Path

from . import room_labels

TOL = 14
MIN_SHARE = 0.3
MIN_SIDE = 8
GROUP_GAP = 0.015   # coloured patches closer than this share of the sheet width belong to one drawing
MIN_DRAWING = 0.2   # a drawing with less legend colour than this share of the largest is an inset (key plan): skipped
# legend words per question key, on top of room_labels.ROOM_TYPES (a legend names categories, e.g. "DORMITORY")
EXTRA = {"bedrooms": r"\bdorm|\bdormitor|\bguest ?room|\bsleeping|\bbunk ?room|\bschlafraum|\bmakuu"}


def legend_entries(img, lines: list[dict]) -> list[dict]:
    """[{'text', 'colour': (b, g, r), 'swatch': [x0, y0, x1, y1], 'line': box}] for legend lines with a colour swatch."""
    import numpy as np

    h, w = img.shape[:2]
    out = []
    for ln in lines:
        if ln.get("vertical"):
            continue
        x0, y0, x1, y1 = ln["box"]
        th = max(4.0, y1 - y0)
        # the swatch sits just left of the text: sample a band there, skipping the gap right next to the letters
        sx0, sx1 = int(max(0, x0 - 2.5 * th)), int(max(0, x0 - 0.6 * th))
        sy0, sy1 = int(y0 + 0.25 * th), int(y1 - 0.25 * th)
        if sx1 - sx0 < 6 or sy1 <= sy0:
            continue
        patch = img[sy0:sy1, sx0:sx1].reshape(-1, 3).astype(int)
        if len(patch) < 20:
            continue
        med = np.median(patch, 0)
        uniform = (np.abs(patch - med).max(1) <= TOL).mean()
        sat = med.max() - med.min()
        if uniform < 0.7 or sat < 12 or med.min() > 245 or med.max() < 60:  # not a solid colour (white, grey, black, text)
            continue
        out.append({"text": ln["text"], "colour": tuple(int(v) for v in med), "swatch": [sx0, sy0, sx1, sy1], "line": ln["box"]})
    return out


def _type_of(text: str) -> str | None:
    t = room_labels.norm(text)
    for key, pat in EXTRA.items():
        if re.search(pat, t):
            return key
    return room_labels.room_type(text)


def count(image: str | Path, lines: list[dict], key: str) -> dict | None:
    """{'rooms': [{'bbox', 'poly', 'area', 'legend'}], 'legends': [entries used]} or None when the sheet's legend has
    no colour for this room type."""
    import cv2
    import numpy as np

    img = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if img is None:
        return None
    entries = legend_entries(img, lines)
    used = [e for e in entries if _type_of(e["text"]) == key]
    if not used:
        return None
    h, w = img.shape[:2]
    keep = np.ones((h, w), bool)
    for e in entries:  # the legend itself (swatches and their text) is not part of the plan
        x0, y0 = e["swatch"][0], min(e["swatch"][1], e["line"][1])
        x1, y1 = e["line"][2], max(e["swatch"][3], e["line"][3])
        keep[max(0, int(y0) - 6):int(y1) + 6, max(0, int(x0) - 6):int(x1) + 6] = False
    # the drawings on the sheet: every legend colour together, grouped by proximity; keep the main ones
    allc = np.zeros((h, w), np.uint8)
    for e in entries:
        allc |= ((np.abs(img.astype(int) - np.array(e["colour"])).max(2) <= TOL) & keep).astype(np.uint8)
    allc = cv2.morphologyEx(allc, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # colour fringes along lines
    g = max(3, int(GROUP_GAP * w))
    gn, glab, gst, _ = cv2.connectedComponentsWithStats(cv2.dilate(allc, np.ones((g, g), np.uint8)), 8)
    if gn > 1:
        colour_in = np.bincount(glab.ravel(), weights=allc.ravel(), minlength=gn)
        main = [i for i in range(1, gn) if colour_in[i] >= MIN_DRAWING * colour_in[1:].max()]
        keep &= np.isin(glab, main)
    rooms = []
    for e in used:
        m = ((np.abs(img.astype(int) - np.array(e["colour"])).max(2) <= TOL) & keep).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 4)
        comps = [i for i in range(1, n) if min(st[i, 2], st[i, 3]) >= MIN_SIDE and st[i, 4] >= 50]
        if not comps:
            continue
        top = max(st[i, 4] for i in comps)
        for i in comps:
            if st[i, 4] < MIN_SHARE * top:
                continue
            x, y, bw, bh, a = (int(v) for v in st[i])
            cs, _ = cv2.findContours((lab[y:y + bh, x:x + bw] == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cs, key=cv2.contourArea)
            poly = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True).reshape(-1, 2) + [x, y]
            rooms.append({"bbox": [x, y, x + bw, y + bh], "poly": poly.tolist(), "area": a, "legend": e["text"]})
    return {"rooms": rooms, "legends": used}
