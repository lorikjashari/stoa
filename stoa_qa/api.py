"""HTTP API + web app.

    uv run uvicorn stoa_qa.api:app --port 8765        ->  http://localhost:8765

/api/*            JSON for the web app (catalog, sheets, ask, 3D scenes, benchmark)
/                 the web app (web/)
/blueprint3d/     the linked 2D/3D viewer
/ask, /questions  earlier contract, still used by the 3D viewer's own question box

Only web/, blueprint3d/ and dataset files are ever served (the repo root holds .env).
"""
from __future__ import annotations

import json
import re
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import catalog, llm, pipeline, sheets
from .core import DATASET

REPO = Path(__file__).resolve().parent.parent
WEB = REPO / "web"
SCENES = REPO / "blueprint3d" / "viewer" / "scenes"

app = FastAPI(title="OQVAR · Make Blueprints Talk", version="0.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=2000)
_scene_lock = threading.Lock()


class Ask(BaseModel):
    id: str | None = None
    path: str | None = None  # legacy: any file of the drawing
    question: str


def _drawing(id_or_path: str) -> catalog.Drawing:
    cat = catalog.catalog()
    d = cat.get(id_or_path) or cat.get(_file_index().get(id_or_path, ""))
    if d is None:
        raise HTTPException(404, f"unknown drawing: {id_or_path}")
    return d


def _file_index() -> dict[str, str]:
    idx = getattr(_file_index, "_cache", None)
    if idx is None:
        idx = {f: d.id for d in catalog.catalog().values() for f in d.files}
        _file_index._cache = idx  # type: ignore[attr-defined]
    return idx


def _file(rel: str) -> Path:
    try:
        return sheets.safe_path(rel)
    except FileNotFoundError:
        raise HTTPException(404, "file not found") from None


# ------------------------------------------------------------------ catalog

@app.get("/api/config")
def config():
    return {"llm": llm.available(), "llm_model": llm.MODEL.split("/")[-1], "drawings": len(catalog.catalog()),
            "questions": sum(len(v) for v in catalog.question_index().values())}


@app.get("/api/drawings")
def drawings(source: str | None = None, q: str | None = None, only_questions: bool = False, offset: int = 0, limit: int = 60):
    return catalog.search(source, q, only_questions, offset, min(limit, 200))


@app.get("/api/drawing")
def drawing(id: str):
    d = _drawing(id)
    out_sheets = []
    for s in d.sheets:
        pages = sheets.pdf_pages(s.path) if s.kind == "pdf" else 1
        for page in range(min(pages, 12)):
            try:
                w, h = sheets.image_size(s.kind, s.path, page)
            except Exception:
                continue
            src = {"image": f"/api/file?path={_q(s.path)}", "pdf": f"/api/render?path={_q(s.path)}&page={page}",
                   "svg": f"/api/svg?path={_q(s.path)}"}[s.kind]
            label = s.label if pages == 1 else f"{s.label} · page {page + 1}"
            out_sheets.append({"kind": "svg" if s.kind == "svg" else "image", "path": s.path, "page": page, "label": label,
                               "src": src, "width": w, "height": h})
    order = {"Plan": 0, "Elevation": 1, "Section": 2}
    out_sheets.sort(key=lambda s: (order.get(s["label"].split(" ")[0], 3), s["label"]))
    qs = catalog.questions_for(d)
    return {**d.summary(), "ask_path": d.ask_path, "files": d.files, "sheets": out_sheets,
            "questions": _suggest(qs), "question_count": len(qs), "credit": catalog.credit_for(d),
            "source_info": catalog.SOURCE_INFO.get(d.source, {}), "scene": _scene_name(d) if d.model else None}


def _q(s: str) -> str:
    from urllib.parse import quote

    return quote(s, safe="")


def _lang(q: str) -> str:
    w = q.lower().split()[:2]
    table = {"how": "en", "what": "en", "which": "en", "on": "en", "wie": "de", "combien": "fr", "quelle": "fr", "quel": "fr",
             "quanti": "it", "quante": "it", "qual": "it", "kuinka": "fi", "hvor": "da", "hvad": "da", "hoeveel": "nl",
             "wat": "nl", "mitu": "et", "kui": "et", "¿cuál": "es", "ile": "pl", "jaka": "pl", "care": "ro"}
    return table.get(w[0] if w else "", "en")


def _suggest(qs: list[dict], limit: int = 40) -> list[dict]:
    """A varied sample of the dataset's questions for this drawing: one per template and language."""
    seen, out = set(), []
    for q in qs:
        key = (re.sub(r"['«„\"].*?['»“”\"]|\d+(\.\d+)?", "#", q["question"]), _lang(q["question"]))
        if key in seen:
            continue
        seen.add(key)
        out.append({"question": q["question"], "type": q["type"], "lang": _lang(q["question"])})
    out.sort(key=lambda q: (q["lang"] != "en", q["lang"]))
    return out[:limit]


# ------------------------------------------------------------------ sheets

@app.get("/api/file")
def file(path: str):
    return FileResponse(_file(path), headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/render")
def render(path: str, page: int = 0):
    _file(path)
    return FileResponse(sheets.render_pdf(path, page), media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/svg")
def svg(path: str):
    _file(path)
    return Response(sheets.clean_svg(path), media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


PLACEHOLDER = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 240"><rect width="360" height="240" fill="#151b26"/>'
               '<g fill="none" stroke="#3a4a66" stroke-width="2"><path d="M110 170 180 70l70 100z"/><path d="M110 170h140"/>'
               '<path d="M180 70v100"/></g><text x="180" y="205" fill="#6b7c99" font-family="sans-serif" font-size="14" '
               'text-anchor="middle">{}</text></svg>')


@app.get("/api/thumb")
def thumb(id: str, w: int = 360):
    d = _drawing(id)
    if d.sheets:
        s = d.sheets[0]
        try:
            return FileResponse(sheets.thumbnail(s.kind, s.path, min(max(w, 120), 720)), media_type="image/jpeg",
                                headers={"Cache-Control": "public, max-age=86400"})
        except Exception:
            pass
    label = "3D model" if d.model else "/".join(d.formats).upper()
    return Response(PLACEHOLDER.format(label), media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/crop")
def crop(path: str, x0: float, y0: float, x1: float, y1: float, kind: str = "image", page: int = 0):
    _file(path)
    return FileResponse(sheets.crop("pdf" if kind == "pdf" else "image", path, (x0, y0, x1, y1), page), media_type="image/jpeg")


# ------------------------------------------------------------------ questions

@app.post("/api/ask")
def ask(req: Ask):
    d = _drawing(req.id or req.path or "")
    if not req.question.strip():
        raise HTTPException(400, "empty question")
    return pipeline.respond(d, req.question)


@app.get("/api/questions")
def questions(id: str):
    return catalog.questions_for(_drawing(id))


# ------------------------------------------------------------------ 3D scenes (built on demand for blueprint3d/viewer)

def _scene_name(d: catalog.Drawing) -> str:
    return Path(d.model).stem if d.model else ""


@app.post("/api/scene")
def scene(req: Ask):
    d = _drawing(req.id or req.path or "")
    if not d.model:
        raise HTTPException(404, "this drawing has no model for a 3D view")
    name = _scene_name(d)
    out = SCENES / f"{name}.json"
    with _scene_lock:
        if not out.exists():
            sys.path.insert(0, str(REPO / "blueprint3d"))
            try:
                if d.model.endswith(".ifc"):
                    import ifc_to_scene

                    sc = ifc_to_scene.build_scene(str(DATASET / d.model))
                else:
                    import dxf_to_scene

                    sc = dxf_to_scene.build_scene(str(DATASET / d.model), dxf_to_scene.DEFAULTS)
            except Exception as e:
                raise HTTPException(500, f"could not build the 3D scene: {type(e).__name__}: {e}") from None
            SCENES.mkdir(parents=True, exist_ok=True)
            # the viewer resolves files from the repo root: store dataset-relative paths the way its own builders do
            src = sc.get("source", {})
            if "file" in src and not str(src["file"]).startswith("data/"):
                src["file"] = d.model
            out.write_text(json.dumps(sc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            index_path = SCENES / "index.json"
            index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
            index = [x for x in index if x["name"] != name] + [{"name": name, "source": sc["source"]["file"], "counts": sc["counts"]}]
            index_path.write_text(json.dumps(sorted(index, key=lambda x: x["name"]), ensure_ascii=False, indent=1), encoding="utf-8")
    return {"name": name, "url": f"/blueprint3d/viewer/?embed=1#{name}"}


# ------------------------------------------------------------------ benchmark

@app.get("/api/benchmark")
def benchmark():
    f = WEB / "benchmark.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"sources": [], "note": "run python -m stoa_qa.report"}


# ------------------------------------------------------------------ legacy contract (blueprint3d viewer)

@app.get("/health")
@app.get("/api/health")
def health():
    return {"ok": True, "dataset": str(DATASET), **config()}


@app.post("/ask")
def ask_legacy(req: Ask):
    r = ask(req)
    return {**r, "evidence": r["overlay"]["items"]}


@app.get("/questions")
def questions_legacy(path: str):
    return catalog.questions_for(_drawing(path))


@app.get("/")
def root():
    landing = WEB / "landing.html"
    return FileResponse(landing) if landing.exists() else RedirectResponse("/app")


@app.get("/app")
def web_app():
    index = WEB / "index.html"
    return FileResponse(index) if index.exists() else RedirectResponse("/blueprint3d/viewer/")


if WEB.is_dir():
    app.mount("/web", StaticFiles(directory=WEB), name="web")
app.mount("/blueprint3d", StaticFiles(directory=REPO / "blueprint3d", html=True), name="viewer")
app.mount("/junction-dataset/junction-dataset", StaticFiles(directory=DATASET), name="dataset")
