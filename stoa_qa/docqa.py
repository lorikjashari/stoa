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

PROMPT = """SHEET TEXT - one printed row per line, cells separated by " | ", each line starts with its id:{ocr_note}
{text}

QUESTION: {question}

Return JSON: {{"kind": "count" | "lookup",
 "rows": [<ids of the lines the answer is based on. For a count: exactly the lines being counted, one per counted item>],
 "answer": <for lookup: the value as printed, without its unit (the unit goes in "unit"); for count: null>,
 "unit": <unit of the answer or null>,{items_field}
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
    return rows_of(words, pg.rect.width), pg.rect.width, pg.rect.height


def rows_of(words: list[tuple], width: float) -> list[dict]:
    """Words or text lines (x0, y0, x1, y1, text) -> printed rows {"id", "text", "bbox"}, split at wide gaps."""
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
            if b[0] - a[2] > max(26 * char, 0.045 * width):
                segments.append(cur)
                cur = []
            cur.append(b)
        segments.append(cur)
        for seg in segments:
            text = seg[0][4]
            for a, b in zip(seg, seg[1:]):
                text += (" | " if b[0] - a[2] > 2.5 * char else " ") + b[4]
            out.append({"id": f"L{len(out) + 1}", "text": text,
                        "bbox": [min(w[0] for w in seg), min(w[1] for w in seg), max(w[2] for w in seg), max(w[3] for w in seg)],
                        "parts": [{"text": w[4], "bbox": list(w[:4])} for w in seg]})
    return out


def raster_layout(image, key: str) -> list[dict]:
    """Printed rows of a raster sheet from its full-resolution OCR (stoa_qa.sheet_ocr), in image pixels.
    Vertical lines (title blocks) stay rows of their own."""
    from . import sheet_ocr

    lines = sheet_ocr.read(image, key) or []
    flat = [(*ln["box"], ln["text"]) for ln in lines if not ln.get("vertical")]
    rows = rows_of(flat, max((ln["box"][2] for ln in lines), default=1.0)) if flat else []
    for ln in lines:
        if ln.get("vertical"):
            rows.append({"id": f"L{len(rows) + 1}", "text": ln["text"], "bbox": ln["box"],
                         "parts": [{"text": ln["text"], "bbox": ln["box"]}]})
    return rows


OCR_NOTE = ("\n(The text was read by OCR from an image of the sheet: spaces are often missing, and inch marks (\") may come out "
            "as ' or ''. Read values with the drawing's conventions, e.g. pipe and duct sizes are in inches. Equipment is "
            "usually named by tag series with a type prefix, e.g. FP-1..FP-5 fuel pumps, CV-1 check valves, P-1 pumps; "
            "when the question names a prefix, e.g. (CV), count only tags with exactly that prefix (not FSCV-1). A size "
            "printed before a pipe, duct, header or valve name, e.g. 16' SUCTION HEADER, is in inches. "
            "Answer for what this sheet shows: when a table gives several floors and a building total, use the row of "
            "this sheet's own floor (its title says which, e.g. SECOND FLOOR PLAN) unless the question asks for the "
            "building. Only answer a count when the text counts exactly the thing asked: a number of dorms or units "
            "is not a number of beds, doors or bedrooms; then return answer null.)")
ITEMS_FIELD = ('\n "items": [<for a count: the exact printed tag or label of each counted item, one string per item, '
               'e.g. ["CV-1", "CV-2"]; for lookup: []>],')


def _key(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def _counted_items(items: list, rows: list[dict]) -> list[tuple[str, dict, list]]:
    """The listed items that really are printed on the sheet, each once: (item, row, box of the printed part)."""
    out, seen = [], set()
    for it in items or []:
        k = _key(str(it))
        if not k or k in seen:
            continue
        for r in rows:
            part = next((pt for pt in r.get("parts", []) if k in _key(pt["text"])), None)
            if part is not None:
                # the item's own share of a part that holds several (OCR reads "CV-1 CV-2" as one line): narrow the box
                t, (x0, y0, x1, y1) = _key(part["text"]), part["bbox"]
                i = t.find(k)
                if len(t) > len(k) and i >= 0:
                    cw = (x1 - x0) / len(t)
                    x0, x1 = x0 + i * cw, x0 + (i + len(k)) * cw
                out.append((str(it), r, [x0, y0, x1, y1]))
                seen.add(k)
                break
    return out


def has_text(rel: str, page: int = 0, min_rows: int = 8) -> bool:
    try:
        return len(layout(rel, page)[0]) >= min_rows
    except Exception:
        return False


def ask(rel: str, question: str, page: int = 0, rows: list[dict] | None = None) -> tuple[Answer, llm.Usage | None]:
    """`rows` given (a raster sheet's OCR rows, raster_layout): boxes are already in sheet pixels, and a count is the
    number of distinct printed items the model lists that are verified on the sheet (OCR lines merge several tags)."""
    ocr = rows is not None
    if rows is None:
        rows, pw, _ = layout(rel, page)
        k = sheets.image_size("pdf", rel, page)[0] / pw  # PDF points -> pixels of the rendered sheet
    else:
        k = 1.0
    text = "\n".join(f"{r['id']}: {r['text']}" for r in rows)
    if len(text) > 60000:
        text = text[:60000]
    try:
        d, usage = llm.chat_json(SYSTEM, PROMPT.format(text=text, question=question, ocr_note=OCR_NOTE if ocr else "",
                                                       items_field=ITEMS_FIELD if ocr else ""), max_tokens=900, timeout=90)
    except llm.LLMError as e:
        return Answer(None, None, "error", 0.0, f"document model unavailable: {e}"), None
    by_id = {r["id"]: r for r in rows}
    picked = []
    for rid in d.get("rows") or []:
        rid = str(rid) if str(rid).startswith("L") else f"L{rid}"
        if rid in by_id and by_id[rid] not in picked:
            picked.append(by_id[rid])
    ev = [Evidence("box", r["id"], r["text"][:80], None,
                   [[round(r["bbox"][0] * k - 4, 1), round(r["bbox"][1] * k - 3, 1)], [round(r["bbox"][2] * k + 4, 1), round(r["bbox"][3] * k + 3, 1)]])
          for r in picked]
    note = str(d.get("note") or "").strip()
    if d.get("kind") == "count" and ocr:
        items = d.get("items") or []
        prefix = re.search(r"\(([A-Za-z]{1,6})\)", question)  # "check valves (CV)": only CV-n tags, never FSCV-n
        if prefix:
            items = [it for it in items if re.match(rf"^{prefix.group(1)}[\s\-#.]*\d", str(it).strip(), re.I)]
        found = _counted_items(items, picked or rows)
        if found:
            ev = [Evidence("box", f"{r['id']}-{i}", it, None, [[round(b[0] - 3, 1), round(b[1] - 3, 1)], [round(b[2] + 3, 1), round(b[3] + 3, 1)]])
                  for i, (it, r, b) in enumerate(found)]
            listed = len({_key(str(x)) for x in items if _key(str(x))})
            miss = f"; {listed - len(found)} listed item(s) not found in the sheet text were not counted" if listed > len(found) else ""
            return Answer(len(found), "count", "doc_count", 0.7,
                          f"Counted {len(found)} printed items on the sheet ({', '.join(it for it, _, _ in found[:8])}"
                          f"{', ...' if len(found) > 8 else ''}){miss}" + (f" · {note}" if note else ""), ev), usage
    if d.get("kind") == "count":
        return Answer(len(picked), "count", "doc_count", 0.7 if picked else 0.4,
                      f"Counted {len(picked)} rows of the sheet's text selected for the question" + (f" · {note}" if note else ""), ev), usage
    value = d.get("answer")
    if isinstance(value, str) and value.strip().lower() in ("", "null", "none"):
        value = None
    if isinstance(value, str) and re.fullmatch(r"-?\d+(\.\d+)?", value.strip()):
        value = float(value) if "." in value else int(value)
    unit = d.get("unit") if isinstance(d.get("unit"), str) else None
    if unit and isinstance(value, str) and _key(unit) and _key(unit) in _key(value):
        unit = None  # the printed value already carries its unit ("26,562 SF"): don't print it twice
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
