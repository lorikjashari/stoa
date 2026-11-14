"""Legends on real drawing sheets: the block that defines what a symbol or a type mark means.

Found from the PDF's own text, never guessed: a heading such as LEGEND, SYMBOLS, ABBREVIATIONS, ROOM LEGEND,
LIFE SAFETY LEGEND or DOOR / WINDOW / WALL TYPES, plus the printed lines under it (or above it, for a
drawing title such as '7 LIFE SAFETY LEGEND 1/8" = 1'-0"', which architects print beneath the view).
Sheets without such a heading (Swiss CAD plans, CubiCasa scans) have no legend, and none is shown.

Coordinates are pixels of the sheet image the app shows (sheets.render_pdf, long side 3000 px), so the web app
can crop the legend straight from that image.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from . import sheets
from .core import DATASET

RENDER_SIDE = 3000  # sheets.render_pdf default
HEADING = re.compile(
    r"^(?P<lead>[\d\s/\"'=.:,-]*?\d?\s*)?"
    r"(?P<title>(?:[A-Z][A-Z &/-]*\s)?LEGEND|SYMBOLS?(?: LEGEND)?|GRAPHIC SYMBOLS|ABBREVIATIONS|KEYNOTES?|"
    r"(?:DOOR|WINDOW|WALL|FRAME|PARTITION|ROOM|FINISH)(?: AND \w+)? TYPES)"
    r"(?P<tail>\s+(?:SCALE:?\s*)?[\d/\"'=\s.:-]*|\s*)$")
SCALE = re.compile(r"\d+\s*/\s*\d+\"?\s*=|=\s*1'|1\s*:\s*\d+|SCALE")
MAX_SIBLINGS = 40


def _lines(rel: str, page: int):
    """(text, bbox) of every printed line, in rendered-image pixels."""
    import pymupdf

    pg = pymupdf.open(sheets.safe_path(rel))[page]
    zoom = RENDER_SIDE / max(pg.rect.width, pg.rect.height)
    rot = pg.rotation_matrix
    lines: dict[tuple[int, int], list] = {}
    for w in pg.get_text("words"):
        lines.setdefault((w[5], w[6]), []).append(w)
    out = []
    for ws in lines.values():
        ws.sort(key=lambda w: w[7])
        r = pymupdf.Rect()
        for w in ws:
            r |= pymupdf.Rect(w[:4]) * rot
        out.append((" ".join(w[4] for w in ws), [r.x0 * zoom, r.y0 * zoom, r.x1 * zoom, r.y1 * zoom]))
    W, H = pg.rect.width * zoom, pg.rect.height * zoom
    return out, W, H


@lru_cache(maxsize=256)
def find(rel: str, page: int = 0) -> tuple[dict, ...]:
    """Every legend block on one PDF page: {title, bbox, rows: [{text, bbox}]}, pixels of the rendered sheet."""
    try:
        lines, W, H = _lines(rel, page)
    except Exception:
        return ()
    found = []
    for text, b in lines:
        t = " ".join(text.upper().split())
        if len(t) > 60:
            continue
        m = HEADING.match(t)
        if not m:
            continue
        # a view title (number bubble, title, scale line under it) sits beneath its view; a plain heading sits above
        h = b[3] - b[1]
        scale_under = any(SCALE.search(tx.upper()) for tx, bb in lines
                          if bb is not b and 0 <= bb[1] - b[3] <= 3 * h and bb[0] < b[2] and bb[2] > b[0] - 0.05 * W)
        below_title = scale_under or bool(SCALE.search(t)) or bool((m.group("lead") or "").strip())
        x0, x1 = b[0] - 0.03 * W, b[0] + 0.26 * W
        y0, y1 = (b[1] - 0.32 * H, b[1] - 2) if below_title else (b[3] + 2, b[3] + 0.32 * H)
        rows = [{"text": tx, "bbox": [round(v, 1) for v in bb]} for tx, bb in lines
                if bb is not b and x0 <= bb[0] <= x1 and y0 <= bb[1] <= y1 and bb[3] <= y1 + 20 and len(tx) < 160
                and bb[0] < 0.88 * W and not SCALE.search(tx.upper())]  # not the title block on the sheet's right edge
        if len(rows) < 3:
            continue
        ux0 = min(r["bbox"][0] for r in rows + [{"bbox": b}]) - 0.02 * W
        uy0 = min(r["bbox"][1] for r in rows + [{"bbox": b}]) - 10
        ux1 = max(r["bbox"][2] for r in rows + [{"bbox": b}]) + 10
        uy1 = max(r["bbox"][3] for r in rows + [{"bbox": b}]) + 10
        box = [max(0.0, ux0), max(0.0, uy0), min(W, ux1), min(H, uy1)]
        found.append({"title": m.group("title").title(), "bbox": [round(v, 1) for v in box], "rows": rows[:80]})
    found.sort(key=lambda f: -len(f["rows"]))
    kept: list[dict] = []
    for f in found:  # several headings over one block (DOOR TYPES / FRAME TYPES / KEYNOTE): keep the fullest
        if all(_overlap(f["bbox"], k["bbox"]) < 0.5 for k in kept):
            kept.append(f)
    return tuple(kept[:4])


def _overlap(a: list[float], b: list[float]) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / small if small > 0 else 0.0


def _pdf_sheets(d) -> list[tuple[str, int]]:
    return [(s.path, s.page) for s in d.sheets if s.kind == "pdf"]


@lru_cache(maxsize=128)
def _siblings(rel: str) -> tuple[str, ...]:
    """Other PDF sheets of the same drawing set (same folder), nearest names first."""
    folder = (DATASET / rel).parent
    try:
        pdfs = sorted(p for p in folder.glob("*.pdf") if p.name != Path(rel).name)
    except OSError:
        return ()
    return tuple(str(p.relative_to(DATASET)).replace("\\", "/") for p in pdfs[:MAX_SIBLINGS])


def for_drawing(d) -> list[dict]:
    """Legends that can explain marks on this drawing: its own sheets first, then the other sheets of its set."""
    own = [(p, pg) for p, pg in _pdf_sheets(d)]
    out = []
    for rel, page in own:
        out += [{**lg, "path": rel, "page": page, "own": True} for lg in find(rel, page)]
    if not out and own:
        for rel in _siblings(own[0][0]):
            out += [{**lg, "path": rel, "page": 0, "own": False} for lg in find(rel, 0)]
            if len(out) >= 3:
                break
    for lg in out:
        w, h = sheets.image_size("pdf", lg["path"], lg["page"])
        lg.update({"width": w, "height": h, "sheet": Path(lg["path"]).stem,
                   "src": f"/api/render?path={_quote(lg['path'])}&page={lg['page']}"})
    return out


def _quote(s: str) -> str:
    from urllib.parse import quote

    return quote(s, safe="")


MARK = re.compile(r"\b[A-Z]{0,3}-?\d{1,3}(?:\.\d)?[A-Z]?\b")  # type marks: B3, A1, 2BD, 1.5, W-12


def match(legends: list[dict], *texts: str) -> tuple[int, int] | None:
    """(legend index, row index) of the legend row naming a type mark from these texts (e.g. door type 'B3').

    Only marks with a digit count: words like DOOR or ROOM appear in every row and would point anywhere."""
    marks = {m.replace("-", "") for t in texts if t for m in MARK.findall(str(t).upper())}
    if not marks:
        return None
    for li, lg in enumerate(legends):
        for ri, r in enumerate(lg["rows"]):
            words = {w.replace("-", "") for w in re.findall(r"[A-Z0-9.-]+", r["text"].upper())}
            if marks & words:
                return li, ri
    return None
