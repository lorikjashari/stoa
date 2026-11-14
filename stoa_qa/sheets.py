"""Sheets: the original drawing as the user sees it, and the mapping from engine coordinates onto it.

- raster sheets (PNG/JPG) are served as they are; PDF sheets are rendered once with PyMuPDF and cached
- swiss_dwellings: DXF metres -> PNG pixels, fitted per floor from the room numbers printed on both
- ifc_models: storey-plan SVGs carry IFC GlobalIds (data-guid), so evidence is highlighted by id
"""
from __future__ import annotations

import hashlib
import io
import re
from functools import lru_cache
from pathlib import Path

from PIL import Image

from .core import DATASET

Image.MAX_IMAGE_PIXELS = None
CACHE = Path(__file__).resolve().parent.parent / "eval_out" / "sheet_cache"
ROOM_NO = re.compile(r"^U?\d+\.\d+$")


def _cache_file(kind: str, key: str, ext: str) -> Path:
    d = CACHE / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / (hashlib.sha1(key.encode()).hexdigest()[:20] + ext)


def safe_path(rel: str) -> Path:
    """Dataset-relative path -> absolute path, refusing anything outside the dataset."""
    p = (DATASET / rel).resolve()
    if not p.is_relative_to(DATASET.resolve()) or not p.is_file():
        raise FileNotFoundError(rel)
    return p


def render_pdf(rel: str, page: int = 0, max_side: int = 3000) -> Path:
    """PDF page -> cached PNG (longest side max_side px)."""
    out = _cache_file("pdf", f"{rel}|{page}|{max_side}", ".png")
    if not out.exists():
        import pymupdf

        doc = pymupdf.open(safe_path(rel))
        pg = doc[page]
        zoom = max_side / max(pg.rect.width, pg.rect.height)
        pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False).save(out)
    return out


def pdf_pages(rel: str) -> int:
    import pymupdf

    return len(pymupdf.open(safe_path(rel)))


def raster_path(kind: str, rel: str, page: int = 0) -> Path:
    return render_pdf(rel, page) if kind == "pdf" else safe_path(rel)


@lru_cache(maxsize=4096)
def image_size(kind: str, rel: str, page: int = 0) -> tuple[int, int]:
    if kind == "svg" or rel.lower().endswith(".svg"):
        head = safe_path(rel).read_text(encoding="utf-8", errors="replace")[:2000]
        m = re.search(r'viewBox="([-\d.\s,]+)"', head)
        if m:
            v = [float(x) for x in re.split(r"[\s,]+", m[1].strip())]
            return round(v[2]), round(v[3])
        return 1000, 1000
    with Image.open(raster_path(kind, rel, page)) as im:
        return im.size


def thumbnail(kind: str, rel: str, width: int = 360) -> Path:
    out = _cache_file("thumb", f"{kind}|{rel}|{width}", ".jpg")
    if not out.exists():
        if kind == "svg" or rel.lower().endswith(".svg"):
            import pymupdf

            pg = pymupdf.open(safe_path(rel))[0]
            zoom = width / max(pg.rect.width, 1)
            im = Image.open(io.BytesIO(pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False).tobytes("png")))
        else:
            im = Image.open(raster_path(kind, rel))
        im = im.convert("RGB")
        im.thumbnail((width, width * 2))
        im.save(out, "JPEG", quality=80)
    return out


def crop(kind: str, rel: str, box: tuple[float, float, float, float], page: int = 0, pad: float = 0.35, size: int = 220) -> Path:
    """Padded crop around an evidence box (image pixels) -> cached JPEG."""
    out = _cache_file("crop", f"{kind}|{rel}|{page}|{[round(v) for v in box]}|{pad}|{size}", ".jpg")
    if not out.exists():
        with Image.open(raster_path(kind, rel, page)) as im:
            x0, y0, x1, y1 = box
            m = max(x1 - x0, y1 - y0) * pad + 12
            c = im.convert("RGB").crop((max(0, x0 - m), max(0, y0 - m), min(im.width, x1 + m), min(im.height, y1 + m)))
            c.thumbnail((size, size))
            c.save(out, "JPEG", quality=85)
    return out


# ------------------------------------------------------------------ swiss: DXF metres -> sheet pixels

@lru_cache(maxsize=512)
def swiss_transform(stem: str) -> tuple[float, float, float] | None:
    """(s, tx, ty) with  px = s*x + tx,  py = -s*y + ty  mapping DXF metres onto the PNG sheet.

    Fitted by least squares on the room numbers, which are printed at the same place in the DXF
    (centre-aligned TEXT) and in the PDF the PNG was rendered from."""
    import pymupdf

    from .swiss import load_floor

    base = DATASET / "data/swiss_dwellings"
    try:
        page = pymupdf.open(base / f"{stem}.pdf")[0]
        w_px, _ = image_size("image", f"data/swiss_dwellings/{stem}.png")
    except Exception:
        return None
    k = w_px / page.rect.width  # PDF points -> PNG pixels
    pdf_pos: dict[str, list[tuple[float, float]]] = {}
    for x0, y0, x1, y1, word, *_ in page.get_text("words"):
        if ROOM_NO.match(word):
            pdf_pos.setdefault(word, []).append(((x0 + x1) / 2 * k, (y0 + y1) / 2 * k))
    dxf_pos: dict[str, list[tuple[float, float]]] = {}
    for t, _, align in load_floor(stem).room_texts:  # the one DXF reader (stoa_qa.swiss)
        if ROOM_NO.match(t):
            dxf_pos.setdefault(t, []).append(align)
    pairs = [(dxf_pos[t][0], pdf_pos[t][0]) for t in dxf_pos if t in pdf_pos and len(dxf_pos[t]) == 1 and len(pdf_pos[t]) == 1]
    if len(pairs) < 2:
        return None
    n = len(pairs)
    mx, my = sum(p[0][0] for p in pairs) / n, sum(p[0][1] for p in pairs) / n
    mX, mY = sum(p[1][0] for p in pairs) / n, sum(p[1][1] for p in pairs) / n
    num = sum((d[0] - mx) * (s[0] - mX) - (d[1] - my) * (s[1] - mY) for d, s in pairs)
    den = sum((d[0] - mx) ** 2 + (d[1] - my) ** 2 for d, _ in pairs)
    if den == 0:
        return None
    s = num / den
    return s, mX - s * mx, mY + s * my


def swiss_to_px(stem: str, pts: list[list[float]]) -> list[list[float]] | None:
    t = swiss_transform(stem)
    if t is None:
        return None
    s, tx, ty = t
    return [[round(s * x + tx, 1), round(-s * y + ty, 1)] for x, y in pts]


# ------------------------------------------------------------------ ifc: which plan shows which element

@lru_cache(maxsize=64)
def svg_guids(rel: str) -> frozenset[str]:
    return frozenset(re.findall(r'data-guid="([^"]+)"', safe_path(rel).read_text(encoding="utf-8", errors="replace")))


def clean_svg(rel: str) -> str:
    """SVG text safe to inline: scripts and event handlers removed."""
    text = safe_path(rel).read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"<script\b.*?</script>", "", text, flags=re.S | re.I)
    text = re.sub(r"\son\w+\s*=\s*(\"[^\"]*\"|'[^']*')", "", text)
    return re.sub(r"^<\?xml[^>]*\?>\s*", "", text)
