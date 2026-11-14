"""HTTP API + web app.

    uv run uvicorn stoa_qa.api:app --port 8765        ->  http://localhost:8765

/api/*            JSON for the web app (catalog, sheets, ask, 3D scenes, benchmark)
/                 the web app (web/)
/blueprint3d/     the linked 2D/3D viewer
/ask, /questions  earlier contract, still used by the 3D viewer's own question box
/api/upload       a new plan image (stored under the uploads folder, stoa_qa.uploads), then asked like any drawing

Only web/, blueprint3d/, dataset files and uploaded plans are ever served (the repo root holds .env).
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from . import catalog, cubicasa, cubicasa_live, graph, llm, pipeline, plan3d, sheets, uploads
from .core import DATASET

REPO = Path(__file__).resolve().parent.parent
WEB = REPO / "web"
SCENES = REPO / "blueprint3d" / "viewer" / "scenes"

app = FastAPI(title="OQVAR · Make Blueprints Talk", version="0.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=2000)


@app.middleware("http")
async def revalidate_pages(request, call_next):
    """Pages and their scripts revalidate on every load (ETag makes it cheap), so an edit never mixes old and new files."""
    response = await call_next(request)
    path = request.url.path
    if path in ("/", "/app") or path.startswith("/web/"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response
_scene_lock = threading.Lock()


class Ask(BaseModel):
    id: str | None = None
    path: str | None = None  # legacy: any file of the drawing
    question: str
    sheet_path: str | None = None  # the sheet on screen (multi-page PDF sets: which page the question is about)
    sheet_page: int = 0


def _drawing(id_or_path: str) -> catalog.Drawing:
    if uploads.is_upload(id_or_path):
        d = uploads.drawing(id_or_path)
        if d is None:
            raise HTTPException(404, f"unknown upload: {id_or_path}")
        return d
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
            "questions": sum(len(v) for v in catalog.question_index().values()), "detector": _detector()}


def _detector() -> dict:
    """Which YOLO runs can be executed here on plans without precomputed boxes (stoa_qa.cubicasa_live)."""
    run = cubicasa.picks().get("yolo_run")
    return {"run": run, "members": list(cubicasa_live.members(run)), "available": cubicasa_live.available_runs(run)}


@app.get("/api/drawings")
def drawings(source: str | None = None, q: str | None = None, only_questions: bool = False, offset: int = 0, limit: int = 60):
    return catalog.search(source, q, only_questions, offset, min(limit, 200))


@app.get("/api/drawing")
def drawing(id: str):
    d = _drawing(id)
    if uploads.is_upload(d.id):
        return _upload_detail(d)
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
            "source_info": catalog.SOURCE_INFO.get(d.source, {}), "scene": _scene_name(d) if d.model else None,
            "model": d.model, "model_note": d.model_note}


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


# ------------------------------------------------------------------ uploads

@app.post("/api/upload")
async def upload(request: Request):
    """Raw image bytes in the body (PNG, JPEG, WebP, TIFF, BMP) -> the drawing record of the stored plan, ready for
    /api/ask. A sheet larger than uploads.WORK_SIDE is stored scaled; the reply then has `scaled_from` = [w, h]."""
    if int(request.headers.get("content-length") or 0) > uploads.MAX_BYTES:
        raise HTTPException(413, f"file is larger than {uploads.MAX_BYTES // (1024 * 1024)} MB")
    data = await request.body()
    try:
        rel = await run_in_threadpool(uploads.save, data)  # image decode + PNG encode: off the event loop
    except uploads.BadUpload as e:
        raise HTTPException(400, f"cannot use this file: {e}") from None
    d = uploads.drawing(rel)
    try:  # OCR the sheet in the background so the first text question doesn't wait ~25 s
        from . import sheet_ocr

        image = sheets.safe_path(rel)
        sheet_ocr.warm(image, cubicasa_live.image_hash(image))
    except Exception:
        pass
    out = _upload_detail(d)
    try:  # say when a very large sheet was stored scaled (uploads.WORK_SIDE)
        import io

        from PIL import Image

        Image.MAX_IMAGE_PIXELS = uploads.MAX_PIXELS
        with Image.open(io.BytesIO(data)) as im:
            if max(im.size) > uploads.WORK_SIDE:
                out["scaled_from"] = list(im.size)
    except Exception:
        pass
    return out


def _upload_detail(d: catalog.Drawing) -> dict:
    s = d.sheets[0]
    w, h = sheets.image_size("image", s.path)
    det = _detector()
    if det["available"]:
        note = f"Counted by the YOLO11 plan detectors run on this image ({' + '.join(det['available'])})."
    else:
        note = ("No plan detector is installed on this server (ultralytics + weights in third_party/weights/), so counts come "
                "from a vision model as estimates, or are refused.")
    return {**d.summary(), "ask_path": d.ask_path, "files": d.files, "upload": True,
            "sheets": [{"kind": "image", "path": s.path, "page": 0, "label": s.label, "src": f"/api/file?path={_q(s.path)}",
                        "width": w, "height": h}],
            "questions": [{"question": q, "type": "count", "lang": "en"} for q in uploads.SUGGESTED], "question_count": 0,
            "credit": {"title": "Your upload", "credit": note, "licence": "", "url": ""},
            "source_info": {"label": "Uploaded plan", "kind": "raster floor plan", "engine": "detection"},
            "scene": _scene_name(d) or None, "model": d.model, "model_note": d.model_note, "detector": det}


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
def crop(path: str, x0: float, y0: float, x1: float, y1: float, kind: str = "image", page: int = 0, size: int = 220):
    _file(path)
    size = max(80, min(int(size), 900))  # larger crops for referenced details
    return FileResponse(sheets.crop("pdf" if kind == "pdf" else "image", path, (x0, y0, x1, y1), page, size=size), media_type="image/jpeg")


# ------------------------------------------------------------------ questions

@app.post("/api/ask")
def ask(req: Ask):
    d = _drawing(req.id or req.path or "")
    if not req.question.strip():
        raise HTTPException(400, "empty question")
    sheet = (req.sheet_path, req.sheet_page) if req.sheet_path and req.sheet_path in d.files else None
    return pipeline.respond(d, req.question, sheet)


@app.get("/api/questions")
def questions(id: str):
    return catalog.questions_for(_drawing(id))


@app.get("/api/graph")
def room_graph(id: str):
    """The drawing's room graph (stoa_qa.graph). Swiss sheets get pixel positions so the app can draw it."""
    d = _drawing(id)
    g = graph.for_path(d.ask_path)
    if g is None:
        raise HTTPException(404, "no room topology for this drawing")
    out = graph.to_json(g)
    out["space"] = None
    if d.source == "swiss_dwellings":
        stem = Path(d.ask_path).stem
        for item in out["nodes"] + out["edges"]:
            c = item.pop("center")
            px = sheets.swiss_to_px(stem, [list(c)]) if c else None
            item["px"] = px[0] if px else None
        out["space"] = "px" if any(n["px"] for n in out["nodes"]) else None
    else:
        for item in out["nodes"] + out["edges"]:
            item.pop("center", None)
    out["counts"] = {k: sum(e["kind"] == k for e in out["edges"]) for k in ("door", "open", "wall", "stair")}
    return out


# ------------------------------------------------------------------ 3D scenes (built on demand for blueprint3d/viewer)

def _scene_name(d: catalog.Drawing) -> str:
    if not d.model:
        return ""
    stem = Path(d.model).stem
    if d.source == "opening_design":  # project models have generic names ("Model.ifc"): prefix the project
        stem = f"{Path(d.model).parts[2]}-{stem}"
    if _detector_scene(d):  # built by stoa_qa.plan3d: a name of its own, so an older network-only scene never shadows it
        stem += "-y"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-")


def _detector_scene(d: catalog.Drawing) -> bool:
    """CubiCasa-style raster plans get the scene whose doors, windows, fixtures and rooms are the answers' detector boxes."""
    return d.model_kind == "raster" and d.source == "cubicasa5k" and plan3d.available()


def _build_scene(d: catalog.Drawing, name: str) -> dict:
    """Scene JSON for a drawing's model, using the same readers that answer questions."""
    if _detector_scene(d):
        return plan3d.for_drawing(d.ask_path, d.model)
    sys.path.insert(0, str(REPO / "blueprint3d"))
    path = str(DATASET / d.model)
    if d.model_kind == "ifc":
        import ifc_to_scene

        return ifc_to_scene.build_scene(path)
    if d.model_kind == "dxf":
        import dxf_to_scene

        return dxf_to_scene.build_scene(path, dxf_to_scene.DEFAULTS)
    if d.model_kind == "mesh":
        import mesh_to_scene

        return mesh_to_scene.build_scene(path, str(SCENES / f"{name}.mesh.bin"))
    if d.model_kind == "raster":
        stem = Path(path).stem
        archive = d.source in catalog.ARCHIVE_SOURCES  # needs the floor-plan check, which only raster_to_scene.build() runs
        if catalog._raster_engine() and archive:
            import raster_to_scene

            try:
                return raster_to_scene.build(path, str(SCENES / "_build"))
            except raster_to_scene.NotAPlan as e:
                raise NotBuildable(e.reason) from None
        if not archive and (catalog._raster_engine() or (catalog.NET_CACHE / f"{stem}.json").exists()):  # torch here, or nothing to run
            import raster_to_scene

            return raster_to_scene.build_scene_model(path)
        return _raster_scene_subprocess(path)
    raise ValueError(f"no 3D builder for {d.model_kind!r}")


class NotBuildable(Exception):
    """The drawing cannot become a 3D scene (not a floor plan, nothing detected): the reason is shown to the user."""


def _raster_scene_subprocess(path: str) -> dict:
    """Run blueprint3d/raster_to_scene.py --json in the torch environment (.venv) when this one has no torch.

    Contract (blueprint3d/raster_to_scene.py): one JSON line on stdout per image; exit 0 built ({"ok": true, "scene": file}),
    2 not a plan ({"ok": false, "kind": "not_a_plan"|"too_little", "reason"}, also <stem>.reject.json), 1 error.
    A reject file is cached so a sheet that is not a plan answers at once the next time."""
    import subprocess

    tmp = SCENES / "_build"
    tmp.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    reject = tmp / f"{stem}.reject.json"
    if reject.exists():
        msg = json.loads(reject.read_text(encoding="utf-8"))
        raise NotBuildable(msg.get("reason") or "this sheet is not a floor plan")
    cmd = [str(catalog.RASTER_PY), "-W", "ignore", str(REPO / "blueprint3d" / "raster_to_scene.py"), path, "-o", str(tmp), "--json"]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(REPO), env=env)
    msg = {}
    for line in (p.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                msg = json.loads(line)
                break
            except ValueError:
                pass
    if msg.get("ok") and Path(msg.get("scene") or tmp / f"{stem}.json").exists():
        out = Path(msg.get("scene") or tmp / f"{stem}.json")
        sc = json.loads(out.read_text(encoding="utf-8"))
        out.unlink(missing_ok=True)
        return sc
    if msg.get("kind") in ("not_a_plan", "too_little"):
        raise NotBuildable(msg.get("reason") or "this sheet is not a floor plan")
    tail = (msg.get("error") or ((p.stderr or "") + (p.stdout or "")).strip().splitlines()[-1:] or ["no output"])
    raise NotBuildable(f"the scan could not be traced ({(tail if isinstance(tail, str) else tail[0])[:200]})")


@app.post("/api/scene")
def scene(req: Ask):
    d = _drawing(req.id or req.path or "")
    if not d.model:
        raise HTTPException(404, "this drawing has no model for a 3D view")
    name = _scene_name(d)
    out = SCENES / f"{name}.json"
    with _scene_lock:
        if not out.exists():
            SCENES.mkdir(parents=True, exist_ok=True)
            try:
                sc = _build_scene(d, name)
            except NotBuildable as e:
                raise HTTPException(422, f"No 3D view for this sheet: {e}") from None
            except Exception as e:
                raise HTTPException(500, f"could not build the 3D scene: {type(e).__name__}: {e}") from None
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
if DATASET.is_dir():  # the app also runs without the dataset, for uploaded plans only
    app.mount("/junction-dataset/junction-dataset", StaticFiles(directory=DATASET), name="dataset")
