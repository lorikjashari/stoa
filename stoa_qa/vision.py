"""Vision fallback: when a drawing has no structured data (scans, photos, unknown plans), a small
vision-language model (Gemma 4) reads the sheet. Answers are marked as estimates and every counted
item comes back with a box so a human can still check what the model looked at.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import llm
from .core import Answer, Evidence

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "eval_out" / "vision_cache"

SYSTEM = ("You read architectural drawings (floor plans, sections, elevations, schedules). "
          "Answer only from what is visible in the image. Reply with a single JSON object and nothing else.")

PROMPT = """Question about this drawing: {question}

Return JSON: {{"answer": <number or short text, or null if the drawing does not show it>,
 "unit": "count" | "m2" | "m" | null,
 "items": [{{"label": <what it is>, "box_2d": [ymin, xmin, ymax, xmax]}}],
 "note": <one short sentence: what you looked at>}}
- For counting questions list EVERY counted item in "items" with a tight box, coordinates normalised to 0-1000.
- For other questions put the region(s) you read the answer from in "items" (at most 5)."""


def ask(image: Path, question: str, size: tuple[int, int], cache_key: str | None = None) -> tuple[Answer, llm.Usage | None]:
    cached = None
    if cache_key:
        import hashlib

        CACHE.mkdir(parents=True, exist_ok=True)
        cached = CACHE / (hashlib.sha1(f"{cache_key}|{question}".encode()).hexdigest()[:20] + ".json")
        if cached.exists():
            d = json.loads(cached.read_text())
            return _to_answer(d, size), None
    try:
        d, usage = llm.chat_json(SYSTEM, PROMPT.format(question=question), image=image, max_tokens=900, timeout=75)
    except llm.LLMError as e:
        return Answer(None, None, "error", 0.0, f"vision model unavailable: {e}"), None
    if cached:
        cached.write_text(json.dumps(d))
    return _to_answer(d, size), usage


def _to_answer(d: dict, size: tuple[int, int]) -> Answer:
    w, h = size
    ev = []
    for i, it in enumerate(d.get("items") or []):
        b = it.get("box_2d") if isinstance(it, dict) else None
        if not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)):
            continue
        y0, x0, y1, x1 = b
        ev.append(Evidence("box", f"vision-{i}", str(it.get("label") or f"item {i + 1}"), None,
                           [[round(x0 / 1000 * w, 1), round(y0 / 1000 * h, 1)], [round(x1 / 1000 * w, 1), round(y1 / 1000 * h, 1)]]))
    value = d.get("answer")
    if isinstance(value, str) and value.strip().lower() in ("", "null", "none", "unknown"):
        value = None
    note = str(d.get("note") or "").strip()
    return Answer(value, d.get("unit") if d.get("unit") in ("count", "m2", "m", "m3") else None, "vision",
                  0.4 if value is not None else 0.0,
                  "AI vision estimate (Gemma 4 read the sheet; no geometry to compute from)" + (f" · {note}" if note else ""), ev)
