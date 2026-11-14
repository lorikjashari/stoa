"""HTTP API: one /ask endpoint for every drawing format, plus helpers for the demo UI.

    uv run uvicorn stoa_qa.api:app --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"path": "data/swiss_dwellings/site1047_bldg2154_floor9740.pdf", "question": "How many doors?"}'

/ask returns the viewer contract used by blueprint3d/viewer (text, method, ids) plus value, unit,
confidence and full evidence. Static files: only blueprint3d/ (viewer + scenes) and the dataset are
served, never the repo root (it holds .env).
"""
from __future__ import annotations

import collections
import json
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .core import DATASET, source_of
from .router import answer

REPO = Path(__file__).resolve().parent.parent
UNIT_TEXT = {"m2": "m²", "m3": "m³", "m": "m", "count": ""}

app = FastAPI(title="Stoa · Make Blueprints Talk", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Ask(BaseModel):
    path: str
    question: str


@lru_cache(maxsize=1)
def _questions() -> dict[str, list[dict]]:
    by_path: dict[str, list[dict]] = collections.defaultdict(list)
    for line in open(DATASET / "questions.jsonl", encoding="utf-8"):
        r = json.loads(line)
        by_path[r["path"]].append({"question": r["question"], "answer": r["answer"], "unit": r["unit"], "type": r["type"]})
    return by_path


def _text(a) -> str:
    if a.value is None:
        return "I can't answer that from this drawing yet."
    unit = UNIT_TEXT.get(a.unit or "", a.unit or "")
    return f"{a.value} {unit}".strip()


@app.get("/health")
def health():
    return {"ok": True, "dataset": str(DATASET), "questions_loaded": sum(len(v) for v in _questions().values())}


@app.post("/ask")
def ask(req: Ask):
    if ".." in req.path or not req.path.startswith("data/"):
        raise HTTPException(400, "path must look like data/<source>/<file>")
    t0 = time.perf_counter()
    a = answer(req.path, req.question)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    gold = next((q for q in _questions().get(req.path, []) if q["question"].strip() == req.question.strip()), None)
    return {
        "text": _text(a),
        "method": a.method,
        "ids": [e.viewer_id for e in a.evidence if e.viewer_id is not None],
        "value": a.value,
        "unit": a.unit,
        "intent": a.intent,
        "confidence": a.confidence,
        "evidence": [e.__dict__ for e in a.evidence],
        "source_file": a.source_file,
        "latency_ms": ms,
        "expected": gold["answer"] if gold else None,  # Stoa's answer when the question is from the dataset
    }


@app.get("/drawings")
def drawings(source: str | None = None, limit: int = 200):
    paths = sorted(p for p in _questions() if source is None or source_of(p) == source)
    return [{"path": p, "source": source_of(p), "questions": len(_questions()[p])} for p in paths[:limit]]


@app.get("/questions")
def questions(path: str):
    return _questions().get(path, [])


@app.get("/")
def root():
    return RedirectResponse("/blueprint3d/viewer/")


app.mount("/blueprint3d", StaticFiles(directory=REPO / "blueprint3d", html=True), name="viewer")
app.mount("/junction-dataset/junction-dataset", StaticFiles(directory=DATASET), name="dataset")
