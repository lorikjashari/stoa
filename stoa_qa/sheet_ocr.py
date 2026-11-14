"""The printed text of a raster sheet (a photo, scan or PNG export of a drawing), read at full resolution.

A drawing sheet carries most of its answers as small text: schedules, room counts, area tables, notes, dimensions,
title block. Scaled down to fit a model (or the whole-sheet OCR pass), that text is a few pixels high and unreadable.
So the sheet is read in overlapping tiles at 2x zoom, and again turned 90 degrees for the vertical text of title
blocks; lines cut by an inner tile edge are dropped (the overlapping neighbour reads them whole) and repeats merged.

The lines become printed rows for stoa_qa.docqa (the same row builder as a PDF's text layer), so a question about a
raster sheet is answered from its text with every used row highlighted, instead of a vision model squinting at a
shrunk image. RapidOCR (ONNX, CPU). Cached per image content under eval_out/ocr/<hash>.sheet.json.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "eval_out" / "ocr"
MIN_CONF = 0.6
VIEW = 1600       # pixels the recogniser sees per tile
REF_SIDE = 4000   # printed text scales with the sheet: a sheet this long (long side, px) has text the recogniser
                  # reads as is; a 2000 px sheet (~9 px text) is zoomed 2x. zoom = max(1, REF_SIDE / long side)
OVERLAP = 0.2
SMALL = 1200      # a sheet whose long side is at most this is read in one pass (scaled to ~2000 px)

_ocr = None
_lock = threading.Lock()


# RapidOCR's default recogniser also knows Chinese: drawing sheets here are in Latin-script languages, so CJK
# characters it emits are misreads of symbols (e.g. "FP-4岁"); strip them
CJK = re.compile(r"[⺀-鿿가-힯＀-￯]+")


def clean(text: str) -> str:
    return " ".join(CJK.sub(" ", text).split())


def _engine():
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    return _ocr


def _starts(n: int, tile: int) -> list[int]:
    if n <= tile:
        return [0]
    step = int(tile * (1 - OVERLAP))
    s = list(range(0, n - tile, step))
    return s + [n - tile]


def _iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _read_tiles(rgb, zoom: float) -> list[dict]:
    """OCR lines of an RGB array in tiles of VIEW / zoom source px, each scaled by zoom; boxes in the array's pixels."""
    import cv2

    h, w = rgb.shape[:2]
    tile = int(VIEW / zoom)
    out = []
    for y0 in _starts(h, tile):
        for x0 in _starts(w, tile):
            x1, y1 = min(w, x0 + tile), min(h, y0 + tile)
            crop = rgb[y0:y1, x0:x1]
            if zoom != 1:
                crop = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
            res, _ = _engine()(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            for quad, text, conf in res or []:
                text = clean(text)
                if float(conf) < MIN_CONF or not text:
                    continue
                xs, ys = [q[0] / zoom for q in quad], [q[1] / zoom for q in quad]
                b = [min(xs), min(ys), max(xs), max(ys)]
                edge = 2
                if (b[0] < edge and x0 > 0) or (b[1] < edge and y0 > 0) or (b[2] > x1 - x0 - edge and x1 < w) \
                        or (b[3] > y1 - y0 - edge and y1 < h):
                    continue  # cut by an inner tile edge: the overlapping tile reads it whole
                out.append({"box": [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0], "text": text.strip(), "conf": float(conf)})
    return out


def _inside(a, b) -> float:
    """Share of box a that lies inside box b."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return w * h / area if area > 0 else 0.0


def _merge(lines: list[dict]) -> list[dict]:
    """Drop repeats of a line read from two overlapping tiles, and fragments of a line (the part of it one tile saw)
    lying inside a longer reading of it: the longest, then most confident reading wins."""
    kept: list[dict] = []
    for ln in sorted(lines, key=lambda r: (-len(r["text"]), -r["conf"])):
        if all(_iou(ln["box"], k["box"]) < 0.5 and _inside(ln["box"], k["box"]) < 0.6 for k in kept):
            kept.append(ln)
    return sorted(kept, key=lambda r: (r["box"][1], r["box"][0]))


def read(image: str | Path, key: str | None = None) -> list[dict] | None:
    """Text lines [{'box': [x0, y0, x1, y1] in image px, 'text', 'conf', 'vertical'?}], or None when OCR is unavailable."""
    f = CACHE / f"{key}.sheet.json" if key else None
    if f is not None and f.exists():
        return [ln for ln in ({**ln, "text": clean(ln["text"])} for ln in json.loads(f.read_text(encoding="utf-8"))) if ln["text"]]
    try:
        import cv2
        import numpy as np
        from PIL import Image

        rgb = np.asarray(Image.open(image).convert("RGB"))
        with _lock:
            h, w = rgb.shape[:2]
            if max(h, w) <= SMALL:
                s = min(3.0, 2000 / max(h, w))
                im = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
                res, _ = _engine()(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
                lines = [{"box": [min(q[0] for q in quad) / s, min(q[1] for q in quad) / s,
                                  max(q[0] for q in quad) / s, max(q[1] for q in quad) / s], "text": clean(t), "conf": float(c)}
                         for quad, t, c in res or [] if float(c) >= MIN_CONF and clean(t)]
            else:
                zoom = max(1.0, REF_SIDE / max(h, w))
                lines = _read_tiles(rgb, zoom)
                # vertical text (title blocks, side labels) reads bottom-to-top: turn the sheet clockwise and read again
                rot = np.ascontiguousarray(np.rot90(rgb, k=-1))
                for ln in _read_tiles(rot, zoom):
                    x0, y0, x1, y1 = ln["box"]
                    if (x1 - x0) < 1.5 * (y1 - y0) and len(ln["text"]) > 2:
                        continue  # not a horizontal line in the turned sheet: the first pass has it
                    # turned (x', y') -> original (x = y', y = h - 1 - x')
                    lines.append({"box": [y0, h - 1 - x1, y1, h - 1 - x0], "text": ln["text"], "conf": ln["conf"], "vertical": True})
    except ImportError:
        return None
    out = [{**ln, "box": [round(v, 1) for v in ln["box"]], "conf": round(ln["conf"], 3)} for ln in _merge(lines)]
    if f is not None:
        CACHE.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def warm(image: str | Path, key: str) -> None:
    """Read a sheet in the background (right after upload), so the first question doesn't wait for OCR."""
    threading.Thread(target=lambda: read(image, key), daemon=True).start()
