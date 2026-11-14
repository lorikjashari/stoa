"""Plan images uploaded through the web app: stored by content hash, answered like a CubiCasa raster plan.

An upload lives at <UPLOADS>/cubicasa5k/<hash>.png (default eval_out/uploads/, gitignored; STOA_UPLOADS overrides) and
is addressed as "uploads/cubicasa5k/<hash>.png". That id is its drawing id, its ask_path (core.source_of reads
"cubicasa5k" from it, so the raster template engine answers) and the path sheets.safe_path resolves, which only
ever reaches files inside the uploads folder.
"""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

from .catalog import Drawing, Sheet

REPO = Path(__file__).resolve().parent.parent
PREFIX = "uploads/"
KIND = "cubicasa5k"           # uploads are read with the raster plan detectors (trained on CubiCasa5K)
MAX_BYTES = 30 * 1024 * 1024
MAX_PIXELS = 60_000_000       # a decompression bomb guard; CubiCasa plans are at most a few megapixels
SUGGESTED = ["How many doors are there?", "How many windows are there?", "How many bedrooms are there?",
             "How many bathrooms are there?", "How many separate WC rooms are there?", "How many kitchens are there?",
             "How many toilet bowls are there?", "How many sinks are there?", "How many showers and bathtubs are there?",
             "How many saunas are there?", "How many balconies are there?", "How many floors are drawn on this plan?"]


class BadUpload(ValueError):
    pass


def root() -> Path:
    return Path(os.environ.get("STOA_UPLOADS") or REPO / "eval_out" / "uploads")


def is_upload(rel: str) -> bool:
    return rel.startswith(PREFIX)


def resolve(rel: str) -> Path:
    """'uploads/...' -> absolute file inside the uploads folder; FileNotFoundError for anything else."""
    base = root().resolve()
    p = (base / rel[len(PREFIX):]).resolve()
    if not is_upload(rel) or not p.is_relative_to(base) or not p.is_file():
        raise FileNotFoundError(rel)
    return p


def save(data: bytes) -> str:
    """Validate and store an image; returns its id. The same image uploaded twice gets the same id."""
    from PIL import Image

    if not data:
        raise BadUpload("empty file")
    if len(data) > MAX_BYTES:
        raise BadUpload(f"file is larger than {MAX_BYTES // (1024 * 1024)} MB")
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            if w * h > MAX_PIXELS:
                raise BadUpload(f"image is too large ({w}x{h} px)")
            im.load()
            img = im.convert("RGB")
    except BadUpload:
        raise
    except Exception:
        raise BadUpload("not an image PIL can read (use PNG or JPEG)") from None
    buf = io.BytesIO()
    img.save(buf, format="PNG")  # re-encoded: no metadata, one format for the detectors and the browser
    png = buf.getvalue()
    # named by the stored file's hash, the same key cubicasa_live caches its boxes under
    rel = f"{PREFIX}{KIND}/{hashlib.sha256(png).hexdigest()[:24]}.png"
    dest = root() / rel[len(PREFIX):]
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(png)
        tmp.replace(dest)
    return rel


def drawing(rel: str) -> Drawing | None:
    from . import plan3d

    try:
        resolve(rel)
    except FileNotFoundError:
        return None
    model = rel if plan3d.available() else None  # a 3D view needs the CubiCasa5K network (walls, room shapes)
    return Drawing(id=rel, source=KIND, title=f"Uploaded plan · {Path(rel).stem[:8]}", ask_path=rel, formats=["png"],
                   files=[rel], sheets=[Sheet("image", rel, "Uploaded plan")], model=model,
                   model_kind="raster" if model else None, model_note=MODEL_NOTE if model else None)


MODEL_NOTE = ("Approximate: walls and room shapes traced by the CubiCasa5K network; doors, windows, fixtures and room types "
              "placed from the detector boxes the answers count. Scale from door width (0.9 m); heights assumed.")
