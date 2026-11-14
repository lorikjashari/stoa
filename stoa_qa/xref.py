"""Cross-sheet references: follow a callout like "6A / A-401" or "SEE 5A/A080" to the detail it points at.

Deterministic, from the PDF text layer (PyMuPDF), no LLM:

  sheet index  every PDF page of a project, keyed by its own sheet number. The number is read from the title
               block (largest sheet-number text in the bottom-right corner of the displayed page), or from the
               file name for one-sheet PDFs ("A-401 - Wall Sections.pdf").
  callouts     detail bubbles: a detail code ("6A", "5", "B2") stacked on a sheet number ("A-401");
               inline notes: "SEE 5A/A080", "SEE SHEET A-501", "DETAIL 3/A-502".
  targets      on the referenced sheet: the view titled with that code (its title bubble, title text and scale
               line; the view is the drawing above the title), else the sheet's zone grid cell (border labels
               1..n and A..D), else the whole sheet.

All boxes are in the page's display space (rotation applied, PDF points), as stoa_qa.docqa uses; multiply by
sheets.image_size(...)[0] / page width for pixels of the rendered sheet.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from .core import DATASET, Answer, Evidence

SHEET = re.compile(r"(?!GL|EL)[A-Z]{1,2}-?(?:\d{2,3}(?:\.\d{1,2})?|\d\.\d{1,2})(?:-?[A-Z])?")  # A-301, A301, A3.1, S-100, A-100-A; not W1 / GL-1 tags
DETAIL = re.compile(r"\d{1,2}[A-Z]?|[A-Z]\d{0,2}")
INLINE = re.compile(r"\b(\d{1,2}[A-Z]?|[A-Z]\d?)\s*/\s*((?!GL|EL)[A-Z]{1,2}-?(?:\d{2,3}(?:\.\d{1,2})?|\d\.\d{1,2}))\b")
SEE_SHEET = re.compile(r"\bSEE\s+(?:SHEET\s+|DWG\.?\s+|DRAWING\s+)?((?!GL|EL)[A-Z]{1,2}-?(?:\d{2,3}(?:\.\d{1,2})?|\d\.\d{1,2}))\b")
FILE_SHEET = re.compile(r"(?:^|Sheet - )([A-Z]{1,2}-\d{1,3}(?:\.\d{1,2})?(?:-[A-Z])?)\s+-\s")
REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "eval_out" / "xref"
MAX_PAGES = 80


def norm(sheet: str) -> str:
    """'A-401' and 'A401' are the same sheet."""
    return sheet.upper().replace("-", "").replace(" ", "")


@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def h(self) -> float:  # text height (rotated glyph boxes: the short side)
        return min(self.y1 - self.y0, self.x1 - self.x0) if len(self.text) > 2 else self.y1 - self.y0

    @property
    def c(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    def box(self) -> list[float]:
        return [round(self.x0, 1), round(self.y0, 1), round(self.x1, 1), round(self.y1, 1)]


@lru_cache(maxsize=256)
def page_words(rel: str, page: int = 0) -> tuple[tuple[Word, ...], float, float]:
    import pymupdf

    from . import sheets

    pg = pymupdf.open(sheets.safe_path(rel))[page]
    out = []
    for w in pg.get_text("words"):
        if w[4].strip():
            r = (pymupdf.Rect(w[:4]) * pg.rotation_matrix).normalize()
            out.append(Word(r.x0, r.y0, r.x1, r.y1, w[4].strip()))
    return tuple(out), pg.rect.width, pg.rect.height


def _in_title_block(w: Word, W: float, H: float) -> bool:
    return w.x0 > 0.8 * W and w.y0 > 0.55 * H


def own_number(rel: str, page: int = 0) -> str | None:
    words, W, H = page_words(rel, page)
    cands = [w for w in words if SHEET.fullmatch(w.text) and _in_title_block(w, W, H) and re.search(r"\d", w.text)]
    if cands:
        best = max(cands, key=lambda w: (w.x1 - w.x0) * (w.y1 - w.y0))
        if best.h >= 1.6 * _median_h(words):  # the sheet number is the biggest text in the title block
            return best.text
    m = FILE_SHEET.search(Path(rel).name)
    return m.group(1) if m else None


def _median_h(words) -> float:
    hs = sorted(w.h for w in words) or [10.0]
    return hs[len(hs) // 2]


def _project(rel: str) -> str:
    parts = Path(rel).parts  # data / opening_design / <project> / ...
    return "/".join(parts[:3])


@lru_cache(maxsize=16)
def index(project: str) -> dict[str, list[dict]]:
    """norm(sheet number) -> [{"path", "page", "sheet"}] for every PDF page of a project (cached on disk)."""
    import pymupdf

    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / (re.sub(r"[^\w.-]+", "_", project) + ".json")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    for f in sorted((DATASET / project).rglob("*.pdf")):
        rel = f.relative_to(DATASET).as_posix()
        try:
            n = min(len(pymupdf.open(f)), MAX_PAGES)
        except Exception:
            continue
        for p in range(n):
            try:
                num = own_number(rel, p)
            except Exception:
                num = None
            if num:
                out.setdefault(norm(num), []).append({"path": rel, "page": p, "sheet": num})
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def find_sheet(from_rel: str, sheet: str) -> dict | None:
    """The page that is sheet `sheet`, preferring the same PDF, then the same folder, then the newest set."""
    hits = index(_project(from_rel)).get(norm(sheet), [])
    if not hits:
        return None
    folder = str(Path(from_rel).parent)
    return sorted(hits, key=lambda h: (h["path"] != from_rel, str(Path(h["path"]).parent) != folder, "Current" not in h["path"],
                                       [-ord(c) for c in h["path"]]))[0]


# ------------------------------------------------------------------ callouts on a sheet

@dataclass
class Callout:
    detail: str | None
    sheet: str
    bbox: list[float]
    kind: str          # bubble | inline | sheet
    text: str = ""


@lru_cache(maxsize=128)
def callouts(rel: str, page: int = 0) -> tuple[Callout, ...]:
    words, W, H = page_words(rel, page)
    out: list[Callout] = []
    used: set[int] = set()
    sheets_ = [w for w in words if SHEET.fullmatch(w.text) and re.search(r"\d", w.text) and not _in_title_block(w, W, H)]
    details = [w for w in words if DETAIL.fullmatch(w.text)]
    for s in sheets_:
        best, bd = None, None
        for d in details:
            if id(d) == id(s):
                continue
            dist = ((d.c[0] - s.c[0]) ** 2 + (d.c[1] - s.c[1]) ** 2) ** 0.5
            if dist <= 2.6 * max(s.h, d.h) and abs(d.h - s.h) <= 0.5 * max(s.h, d.h) and (bd is None or dist < bd):
                best, bd = d, dist
        if best is not None:
            box = [min(s.x0, best.x0), min(s.y0, best.y0), max(s.x1, best.x1), max(s.y1, best.y1)]
            out.append(Callout(best.text, s.text, [round(v, 1) for v in box], "bubble", f"{best.text}/{s.text}"))
            used.add(id(s))
    from .docqa import layout  # printed rows of the page, already in display space

    rows, _, _ = layout(rel, page)
    for r in rows:
        t = r["text"].upper()
        for m in INLINE.finditer(t):
            if SHEET.fullmatch(m.group(2)) and re.search(r"\d", m.group(2)):
                out.append(Callout(m.group(1), m.group(2), r["bbox"], "inline", r["text"][:120]))
        for m in SEE_SHEET.finditer(t):
            if not INLINE.search(t[m.start():m.end() + 12]):
                out.append(Callout(None, m.group(1), r["bbox"], "sheet", r["text"][:120]))
    return tuple(out)


# ------------------------------------------------------------------ targets on the referenced sheet

def _zone_grid(words, W, H):
    """Border zone labels: digits along the top/bottom edge (columns), letters along the left/right edge (rows)."""
    cols = {w.text: w.c[0] for w in words if re.fullmatch(r"\d{1,2}", w.text) and (w.c[1] < 0.06 * H or w.c[1] > 0.94 * H)}
    rows = {w.text: w.c[1] for w in words if re.fullmatch(r"[A-H]", w.text) and (w.c[0] < 0.06 * W or w.c[0] > 0.94 * W)}
    return (cols, rows) if len(cols) >= 3 and len(rows) >= 2 else None


def _band(pos: dict[str, float], key: str, lo: float, hi: float) -> tuple[float, float] | None:
    if key not in pos:
        return None
    vals = sorted(pos.values())
    i = vals.index(pos[key])
    left = (vals[i - 1] + vals[i]) / 2 if i > 0 else max(lo, vals[i] - (vals[1] - vals[0]) / 2)
    right = (vals[i] + vals[i + 1]) / 2 if i + 1 < len(vals) else min(hi, vals[i] + (vals[-1] - vals[-2]) / 2)
    return left, right


def view_titles(rel: str, page: int = 0) -> list[dict]:
    """View titles: a detail code with title text on its right and a scale line just below that text."""
    words, W, H = page_words(rel, page)
    out = []
    for d in words:
        if not DETAIL.fullmatch(d.text) or _in_title_block(d, W, H):
            continue
        near = sorted([w for w in words if w is not d and abs(w.c[1] - d.c[1]) <= 0.9 * d.h and 0 <= w.x0 - d.x1 <= 4 * d.h],
                      key=lambda w: (w.c[1], w.x0))
        first = next((w for w in near if re.match(r"[A-Za-z]", w.text)), None)  # titles start with a word, notes with numbers
        if first is None:
            continue
        # the next view's title bubble on the same row ends this title
        stop = min([o.x0 for o in words if o is not d and DETAIL.fullmatch(o.text) and o.x0 > d.x1
                    and abs(o.c[1] - d.c[1]) <= 0.6 * d.h and abs(o.h - d.h) <= 0.3 * d.h] + [float("inf")])
        line = sorted([w for w in words if abs(w.c[1] - first.c[1]) <= 0.35 * first.h and w.x0 >= first.x0 - 1
                       and w.x0 - d.x1 <= 40 * d.h and w.x1 <= stop], key=lambda w: w.x0)
        title = []
        for w in line:  # contiguous words of the title, left to right
            if title and w.x0 - title[-1].x1 > 3 * d.h:
                break
            title.append(w)
        text = " ".join(w.text for w in title)
        if len(re.findall(r"[A-Za-z]", text)) < 4 or not title:
            continue
        below = [w for w in words if title[0].x0 - 2 * d.h <= w.x0 <= title[-1].x1 and 0 < w.y0 - title[0].y1 + 0.3 * first.h <= 2.2 * d.h
                 and w not in title]
        scale = " ".join(w.text for w in sorted(below, key=lambda w: w.x0))
        if not re.search(r"\d+/\d+\"\s*=|=\s*1'|SCALE|\bNTS\b|N\.T\.S|1\s*:\s*\d+", scale, re.I):
            continue
        box = [min(d.x0, title[0].x0), min(d.y0, title[0].y0), max(w.x1 for w in title), max([w.y1 for w in below] + [d.y1])]
        out.append({"detail": d.text, "title": text, "scale": scale, "bbox": [round(v, 1) for v in box]})
    return out


def locate(rel: str, page: int, detail: str | None) -> dict:
    """Where on the sheet the referenced detail is: {"bbox", "how", "title"}."""
    words, W, H = page_words(rel, page)
    if detail is None:
        return {"bbox": [0, 0, round(W, 1), round(H, 1)], "how": "the whole sheet (the note names a sheet, not a detail)", "title": None}
    titles = view_titles(rel, page)
    grid = _zone_grid(words, W, H)
    t = next((v for v in titles if v["detail"].upper() == detail.upper()), None)
    if t is not None:
        x0, ty0, x1, y1 = t["bbox"]
        if grid:  # the view's column(s) of the zone grid, above its title, up to the next title in that column
            cols, rows = grid
            m = re.fullmatch(r"(\d{1,2})([A-H])?", detail.upper())
            band = _band(cols, m.group(1), 0, W) if m else None
            if band:
                x0, x1 = min(x0, band[0]), max(x1, band[1])
        above = [v["bbox"][3] for v in titles if v is not t and v["bbox"][3] < ty0 and v["bbox"][0] < x1 and v["bbox"][2] > x0]
        top = max(above) + 4 if above else (min(grid[1].values()) - 0.5 * (sorted(grid[1].values())[1] - sorted(grid[1].values())[0]) if grid else 0.04 * H)
        return {"bbox": [round(x0 - 6, 1), round(max(top, 0), 1), round(x1 + 6, 1), round(y1 + 6, 1)],
                "how": f"view titled '{t['title']}' ({t['scale']})", "title": t["title"]}
    if grid:
        cols, rows = grid
        m = re.fullmatch(r"(\d{1,2})([A-H])", detail.upper())
        if m:
            cb, rb = _band(cols, m.group(1), 0, W), _band(rows, m.group(2), 0, H)
            if cb and rb:
                return {"bbox": [round(cb[0], 1), round(rb[0], 1), round(cb[1], 1), round(rb[1], 1)],
                        "how": f"zone {m.group(1)}{m.group(2)} of the sheet's border grid", "title": None}
    return {"bbox": [0, 0, round(W, 1), round(H, 1)], "how": f"detail {detail} is not titled on this sheet: showing the whole sheet", "title": None}


def resolve(from_rel: str, c: Callout) -> dict | None:
    hit = find_sheet(from_rel, c.sheet)
    if hit is None:
        return None
    where = locate(hit["path"], hit["page"], c.detail)
    return {"sheet": hit["sheet"], "path": hit["path"], "page": hit["page"], **where}


# ------------------------------------------------------------------ questions

REF_Q = re.compile(r"\b(?:detail|section|callout|reference|refers?|referenced|cross[- ]?ref|see sheet|which sheets?|other sheets?|"
                   r"where (?:is|are|can i find)|show me)\b", re.I)
ASKED = re.compile(r"\b(?:detail|section|elevation|view)?\s*(\d{1,2}[A-Z]?|[A-Z]\d?)\s*(?:/|\bon\b|\bof\b|\bat\b)\s*(?:sheet\s+)?([A-Z]{1,2}-?\d{1,3}(?:\.\d{1,2})?)\b", re.I)
ASKED_SHEET = re.compile(r"\bsheet\s+([A-Z]{1,2}-?\d{1,3}(?:\.\d{1,2})?)\b", re.I)


def intent_of(question: str) -> str | None:
    if ASKED.search(question):
        return "follow"
    if REF_Q.search(question) and re.search(r"refer|callout|cross|which sheets|other sheets|point", question, re.I):
        return "list"
    return None


def _px(rel: str, page: int, box: list[float]) -> list[list[float]]:
    from . import sheets

    _, W, _ = page_words(rel, page)
    k = sheets.image_size("pdf", rel, page)[0] / W
    return [[round(box[0] * k - 4, 1), round(box[1] * k - 4, 1)], [round(box[2] * k + 4, 1), round(box[3] * k + 4, 1)]]


def answer(path: str, question: str, page: int = 0) -> Answer | None:
    """Reference questions on a PDF sheet. The answer carries `references` for the side-by-side view."""
    intent = intent_of(question)
    if intent is None or not path.lower().endswith(".pdf"):
        return None
    here = own_number(path, page) or Path(path).stem
    cs = list(callouts(path, page))
    pairs = []
    if intent == "follow":
        m = ASKED.search(question)
        detail, sheet = m.group(1).upper(), m.group(2).upper()
        mine = [c for c in cs if norm(c.sheet) == norm(sheet) and (c.detail or "").upper() == detail]
        target = resolve(path, Callout(detail, sheet, [0, 0, 0, 0], "asked"))
        if target is None:
            return Answer(None, None, "xref_follow", 0.0, f"Sheet {sheet} is not in this project's drawing set (or its number is not readable text).",
                          source_file=path)
        pairs.append({"callout": mine[0] if mine else None, "target": target, "detail": detail, "sheet": sheet})
        text = f"Detail {detail} on sheet {target['sheet']}" + (f": {target['title']}" if target["title"] else "")
        text += f"\nFound by: {target['how']}"
        if mine:
            text += f"\nCalled out {len(mine)} time(s) on this sheet ({here})"
        ev = [Evidence("box", f"callout-{i}", f"callout {c.text or (c.detail or '') + '/' + c.sheet}", None, _px(path, page, c.bbox))
              for i, c in enumerate(mine)]
        a = Answer(text, None, "xref_follow", 0.9 if target["title"] or "zone" in target["how"] else 0.6,
                   f"Read the callout text on {here}, found sheet {target['sheet']} in the project's sheet index, then {target['how']}.", ev, path)
    else:
        if not cs:
            return Answer("No callouts to other sheets were found in this sheet's text.", None, "xref_list", 0.7,
                          "Searched the sheet's text for detail bubbles (code over sheet number) and SEE notes.", [], path)
        groups: dict[str, list[Callout]] = {}
        for c in cs:
            groups.setdefault(c.sheet, []).append(c)
        lines = []
        for sheet, items in sorted(groups.items(), key=lambda kv: norm(kv[0])):
            target = find_sheet(path, sheet)
            codes = sorted({c.detail for c in items if c.detail})
            lines.append(f"{sheet}: {len(items)} callout(s)" + (f" — details {', '.join(codes)}" if codes else "")
                         + ("" if target else " (sheet not in this set)"))
            for c in items[:4]:
                t = resolve(path, c) if target else None
                if t is not None and len(pairs) < 24:
                    pairs.append({"callout": c, "target": t, "detail": c.detail, "sheet": sheet})
        text = f"{len(cs)} callouts to {len(groups)} sheet(s)\n" + "\n".join(lines)
        ev = [Evidence("box", f"callout-{i}", f"→ {c.text or c.sheet}", None, _px(path, page, c.bbox)) for i, c in enumerate(cs[:200])]
        a = Answer(text, None, "xref_list", 0.85, f"Callouts read from the text of {here}; each resolved through the project's sheet index.", ev, path)
    a.references = [_ref_json(path, page, p) for p in pairs]  # for the side-by-side view (pipeline.respond passes it on)
    return a


def _ref_json(path: str, page: int, p: dict) -> dict:
    c, t = p["callout"], p["target"]
    return {"label": f"{(p['detail'] + ' / ') if p['detail'] else ''}{p['sheet']}",
            "from": {"path": path, "page": page, "box": sum(_px(path, page, c.bbox), []) if c else None, "text": c.text if c else None},
            "to": {"path": t["path"], "page": t["page"], "sheet": t["sheet"], "box": sum(_px(t["path"], t["page"], t["bbox"]), []),
                   "how": t["how"], "title": t["title"]}}


def as_dict(c: Callout) -> dict:
    return asdict(c)
