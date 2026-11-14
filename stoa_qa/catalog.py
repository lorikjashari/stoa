"""Drawing catalog: groups the dataset's files into drawings with sheets, formats, questions and credits."""
from __future__ import annotations

import collections
import json
import os
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from .core import DATASET

IMAGE_EXT = {".png", ".jpg", ".jpeg"}
SOURCE_INFO = {
    "swiss_dwellings": {"label": "Swiss dwellings", "kind": "CAD floor plans (DXF + PDF + PNG)", "engine": "geometry"},
    "ifc_models": {"label": "IFC models", "kind": "BIM models with rendered storey plans", "engine": "bim"},
    "cubicasa5k": {"label": "CubiCasa5K", "kind": "Finnish raster floor plans", "engine": "vision"},
    "ifc_bench": {"label": "IFC-Bench", "kind": "BIM models, free-text questions", "engine": "bim"},
    "opening_design": {"label": "Opening Design", "kind": "Real drawing sets (PDF, DWG, IFC)", "engine": "document"},
    "habs": {"label": "HABS", "kind": "US historic measured drawings", "engine": "vision"},
    "finna": {"label": "Finna", "kind": "Helsinki archive drawings", "engine": "vision"},
    "europeana": {"label": "Europeana", "kind": "European archive drawings", "engine": "vision"},
    "wikimedia": {"label": "Wikimedia", "kind": "Floor plans from Wikimedia Commons", "engine": "vision"},
}
SOURCE_ORDER = list(SOURCE_INFO)


@dataclass
class Sheet:
    kind: str            # image | svg | pdf
    path: str            # dataset-relative file
    label: str
    page: int = 0


@dataclass
class Drawing:
    id: str              # stable id = primary dataset-relative path
    source: str
    title: str
    ask_path: str        # path handed to the answer engine
    formats: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    sheets: list[Sheet] = field(default_factory=list)
    n_questions: int = 0
    model: str | None = None   # .ifc / .dxf usable for the 3D view

    def summary(self) -> dict:
        return {"id": self.id, "source": self.source, "title": self.title, "formats": self.formats,
                "n_questions": self.n_questions, "has_sheet": bool(self.sheets), "has_3d": self.model is not None}

    def detail(self) -> dict:
        return {**asdict(self), "has_3d": self.model is not None}


def _group_key(source: str, rel: Path) -> str:
    stem = rel.stem
    if source == "ifc_models":
        stem = re.sub(r"_(plan|elevation|section)(-.*)?$|_model$", "", stem)
        return stem
    if source == "opening_design":
        return str(rel.with_suffix(""))  # keep folders: same sheet name appears in several projects
    return stem


def _pretty(source: str, key: str) -> str:
    name = Path(key).name
    if source == "swiss_dwellings":
        m = re.match(r"site(\d+)_bldg(\d+)_floor(\d+)", name)
        return f"Site {m[1]} · Building {m[2]} · Floor {m[3]}" if m else name
    if source == "cubicasa5k":
        style, _, num = name.rpartition("-")
        return f"Plan {num} · {style.replace('_', ' ')}"
    if source == "opening_design":
        parts = Path(key).parts
        return f"{parts[1].replace('_', ' ')} · {name}" if len(parts) > 2 else name
    return name.replace("_", " ")


def _sheet_label(name: str) -> str:
    m = re.search(r"_(plan|elevation|section)(?:-(\d+))?-?(.*)$", Path(name).stem)
    if not m:
        return "Sheet"
    tail = m[3].replace("-", " ").strip()
    return f"{m[1].title()}{(' ' + m[2]) if m[2] else ''}{(' · ' + tail) if tail else ''}"


@lru_cache(maxsize=1)
def question_index() -> dict[str, list[dict]]:
    by_path: dict[str, list[dict]] = collections.defaultdict(list)
    with open(DATASET / "questions.jsonl", encoding="utf-8") as fp:
        for line in fp:
            r = json.loads(line)
            by_path[r["path"]].append({"question": r["question"], "answer": r["answer"], "unit": r["unit"], "type": r["type"]})
    return by_path


@lru_cache(maxsize=1)
def catalog() -> dict[str, Drawing]:
    qidx = question_index()
    groups: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    data = DATASET / "data"
    for source in sorted(os.listdir(data)):
        if not (data / source).is_dir():
            continue
        for d, _, files in os.walk(data / source):
            for f in files:
                if f == "LICENSE.md":
                    continue
                rel = (Path(d) / f).relative_to(DATASET)
                groups[(source, _group_key(source, rel.relative_to("data") if source == "opening_design" else rel))].append(rel.as_posix())

    out: dict[str, Drawing] = {}
    for (source, key), files in groups.items():
        files.sort()
        by_ext = collections.defaultdict(list)
        for f in files:
            by_ext[Path(f).suffix.lower()].append(f)
        sheets: list[Sheet] = []
        if source == "ifc_models":
            for f in by_ext[".svg"]:
                sheets.append(Sheet("svg", f, _sheet_label(f)))
            if not sheets:
                sheets += [Sheet("pdf", f, _sheet_label(f)) for f in by_ext[".pdf"]]
                sheets += [Sheet("image", f, _sheet_label(f)) for f in by_ext[".png"]]
        else:
            imgs = [f for e in IMAGE_EXT for f in by_ext[e]]
            if imgs:
                sheets.append(Sheet("image", imgs[0], "Sheet"))
            elif by_ext[".pdf"]:
                sheets.append(Sheet("pdf", by_ext[".pdf"][0], "Sheet"))
            elif by_ext[".svg"]:
                sheets.append(Sheet("image", by_ext[".svg"][0], "Sheet"))
        model = (by_ext[".ifc"] or by_ext[".dxf"] or [None])[0]
        if source == "opening_design" and model and model.endswith(".dxf"):
            model = None  # layer names differ from the Swiss convention: no 3D builder yet
        with_q = [f for f in files if f in qidx]
        primary = (by_ext[".ifc"] or by_ext[".pdf"] or [s.path for s in sheets] or files)[0] if source != "cubicasa5k" else files[0]
        ask = with_q[0] if with_q else primary
        if source == "ifc_models" and by_ext[".ifc"]:
            ask = by_ext[".ifc"][0]
        out[primary] = Drawing(
            id=primary, source=source, title=_pretty(source, key), ask_path=ask,
            formats=sorted({Path(f).suffix.lower().lstrip(".") for f in files}), files=files, sheets=sheets,
            n_questions=sum(len(qidx.get(f, [])) for f in files), model=model)
    return out


def questions_for(d: Drawing) -> list[dict]:
    qidx = question_index()
    seen, out = set(), []
    for f in d.files:
        for q in qidx.get(f, []):
            if q["question"] not in seen:
                seen.add(q["question"])
                out.append({**q, "path": f})
    return out


def search(source: str | None = None, q: str | None = None, only_questions: bool = False,
           offset: int = 0, limit: int = 60) -> dict:
    items = list(catalog().values())
    facets = collections.Counter(d.source for d in items)
    if source:
        items = [d for d in items if d.source == source]
    if only_questions:
        items = [d for d in items if d.n_questions]
    if q:
        terms = q.lower().split()
        items = [d for d in items if all(t in (d.title + " " + d.id).lower() for t in terms)]
    items.sort(key=lambda d: (SOURCE_ORDER.index(d.source) if d.source in SOURCE_ORDER else 99, -min(d.n_questions, 1), d.id))
    return {"total": len(items), "items": [d.summary() for d in items[offset: offset + limit]],
            "facets": [{"source": s, **SOURCE_INFO.get(s, {"label": s}), "count": facets.get(s, 0)} for s in SOURCE_ORDER if facets.get(s)]}


@lru_cache(maxsize=1)
def credits() -> dict[str, dict]:
    """file name -> {title, licence, credit, url} from data/<source>/LICENSE.md tables; '<source>/*' = folder default."""
    out: dict[str, dict] = {}
    data = DATASET / "data"
    for source in os.listdir(data):
        lic = data / source / "LICENSE.md"
        if not lic.is_file():
            continue
        text = lic.read_text(encoding="utf-8")
        head = " ".join(l.strip() for l in text.split("\n\n")[1].splitlines()) if "\n\n" in text else ""
        out[f"{source}/*"] = {"credit": head[:400], "licence": "", "title": "", "url": ""}
        for line in text.splitlines():
            if not line.startswith("| ") or line.startswith(("| file", "|---")):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
            if len(cells) >= 3:
                out[f"{source}/{cells[0]}"] = {"title": cells[1], "licence": cells[2],
                                               "credit": cells[3] if len(cells) > 3 else "", "url": cells[4] if len(cells) > 4 else ""}
    return out


def credit_for(d: Drawing) -> dict:
    c = credits()
    for f in d.files:
        rel = Path(f).relative_to("data")
        hit = c.get(f"{d.source}/{rel.relative_to(d.source).as_posix()}") or c.get(f"{d.source}/{rel.name}")
        if hit:
            return hit
    return c.get(f"{d.source}/*", {})
