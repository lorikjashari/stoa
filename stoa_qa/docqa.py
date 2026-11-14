"""Document QA for PDF sheets with real text (schedules, title blocks, notes).

The sheet's words are rebuilt into printed rows with their positions. Gemma picks the rows that
answer the question; a count is the number of rows it picked (counted by us, not by the model), and
every picked row is highlighted on the sheet.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from . import llm, sheets
from .core import Answer, Evidence

SYSTEM = ("You answer questions about a construction drawing sheet using only its extracted text. "
          "Reply with a single JSON object and nothing else.")

PROMPT = """SHEET TEXT - one printed row per line, cells separated by " | ", each line starts with its id:
{text}

QUESTION: {question}

Return JSON: {{"kind": "count" | "lookup",
 "rows": [<ids of the lines the answer is based on. For a count: exactly the lines being counted, one per counted item>],
 "answer": <for lookup: the value exactly as printed; for count: null>,
 "unit": <unit of the answer or null>,
 "note": <one short sentence>}}
If the sheet text does not contain the answer, return {{"kind": "lookup", "rows": [], "answer": null, "unit": null, "note": "<why>"}}."""


@lru_cache(maxsize=64)
def layout(rel: str, page: int = 0) -> tuple[list[dict], float, float]:
    """Printed rows of a PDF page: [{"id": "L12", "text": "101.A | A1 | 3'-0\\" ...", "bbox": [x0, y0, x1, y1]}], page width, height."""
    import pymupdf

    pg = pymupdf.open(sheets.safe_path(rel))[page]
    words = []
    for w in pg.get_text("words"):
        if w[4].strip():
            r = (pymupdf.Rect(w[:4]) * pg.rotation_matrix).normalize()  # word boxes come unrotated: put them in display space
            words.append((r.x0, r.y0, r.x1, r.y1, w[4]))
    if not words:
        return [], pg.rect.width, pg.rect.height
    if pg.rotation in (90, 270):  # sideways glyph boxes: the short side is the text height
        pass
    heights = sorted(min(w[3] - w[1], w[2] - w[0]) if len(w[4]) > 2 else w[3] - w[1] for w in words)
    tol = heights[len(heights) // 2] * 0.55
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows: list[list] = []
    for w in words:
        yc = (w[1] + w[3]) / 2
        if rows and abs(yc - rows[-1][0]) <= tol:
            rows[-1][1].append(w)
            rows[-1][0] += (yc - rows[-1][0]) / len(rows[-1][1])
        else:
            rows.append([yc, [w]])
    out = []
    for _, ws in rows:
        ws.sort(key=lambda w: w[0])
        char = sum(w[2] - w[0] for w in ws) / max(1, sum(len(w[4]) for w in ws))
        # a printed row can cross unrelated parts of the sheet (a schedule next to notes): split at wide gaps
        segments, cur = [], [ws[0]]
        for a, b in zip(ws, ws[1:]):
            if b[0] - a[2] > max(26 * char, 0.045 * pg.rect.width):
                segments.append(cur)
                cur = []
            cur.append(b)
        segments.append(cur)
        for seg in segments:
            text = seg[0][4]
            for a, b in zip(seg, seg[1:]):
                text += (" | " if b[0] - a[2] > 2.5 * char else " ") + b[4]
            out.append({"id": f"L{len(out) + 1}", "text": text,
                        "bbox": [min(w[0] for w in seg), min(w[1] for w in seg), max(w[2] for w in seg), max(w[3] for w in seg)]})
    return out, pg.rect.width, pg.rect.height


def has_text(rel: str, page: int = 0, min_rows: int = 8) -> bool:
    try:
        return len(layout(rel, page)[0]) >= min_rows
    except Exception:
        return False


def ask(rel: str, question: str, page: int = 0) -> tuple[Answer, llm.Usage | None]:
    rows, pw, _ = layout(rel, page)
    text = "\n".join(f"{r['id']}: {r['text']}" for r in rows)
    if len(text) > 60000:
        text = text[:60000]
    try:
        d, usage = llm.chat_json(SYSTEM, PROMPT.format(text=text, question=question), max_tokens=900, timeout=90)
    except llm.LLMError as e:
        return Answer(None, None, "error", 0.0, f"document model unavailable: {e}"), None
    by_id = {r["id"]: r for r in rows}
    picked = []
    for rid in d.get("rows") or []:
        rid = str(rid) if str(rid).startswith("L") else f"L{rid}"
        if rid in by_id and by_id[rid] not in picked:
            picked.append(by_id[rid])
    k = sheets.image_size("pdf", rel, page)[0] / pw  # PDF points -> pixels of the rendered sheet
    ev = [Evidence("box", r["id"], r["text"][:80], None,
                   [[round(r["bbox"][0] * k - 4, 1), round(r["bbox"][1] * k - 3, 1)], [round(r["bbox"][2] * k + 4, 1), round(r["bbox"][3] * k + 3, 1)]])
          for r in picked]
    note = str(d.get("note") or "").strip()
    if d.get("kind") == "count":
        return Answer(len(picked), "count", "doc_count", 0.7 if picked else 0.4,
                      f"Counted {len(picked)} rows of the sheet's text selected for the question" + (f" · {note}" if note else ""), ev), usage
    value = d.get("answer")
    if isinstance(value, str) and value.strip().lower() in ("", "null", "none"):
        value = None
    if isinstance(value, str) and re.fullmatch(r"-?\d+(\.\d+)?", value.strip()):
        value = float(value) if "." in value else int(value)
    unit = d.get("unit") if isinstance(d.get("unit"), str) else None
    return Answer(value, unit, "doc_lookup", 0.7 if value is not None and picked else (0.4 if value is not None else 0.0),
                  ("Read from the sheet's text" if value is not None else "The sheet's text does not contain the answer") + (f" · {note}" if note else ""),
                  ev), usage


if __name__ == "__main__":  # quick accuracy check on a sheet's dataset questions:  python -m stoa_qa.docqa "<dataset path>" [limit]
    import sys

    from .catalog import question_index

    rel = sys.argv[1]
    qs = question_index().get(rel, [])[: int(sys.argv[2]) if len(sys.argv) > 2 else 999]
    ok = n = 0
    cost = 0.0
    for q in qs:
        a, u = ask(rel, q["question"])
        cost += u.cost_usd if u else 0
        good = None
        if q["type"] == "count":
            good = a.value == q["answer"]
            n += 1
            ok += bool(good)
        print(json.dumps({"q": q["question"][-90:], "gt": q["answer"], "pred": a.value, "ok": good, "rows": len(a.evidence)}, ensure_ascii=False), flush=True)
    print(f"count questions correct: {ok}/{n} · total cost ${cost:.4f}")
