"""Builds a tiny synthetic dataset (no licensed drawings) and points stoa_qa at it before it is imported."""
import json
import math
import os
import tempfile
from pathlib import Path

import ezdxf

ROOT = Path(tempfile.mkdtemp(prefix="stoa-test-"))
STEM = "site1_bldg1_floor1"


def _rect(msp, layer, x0, y0, x1, y1, closed=True, five=False):
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if five:  # door leaf in the dataset: 5 points, first repeated
        pts.append((x0, y0))
    return msp.add_lwpolyline(pts, close=closed, dxfattribs={"layer": layer})


def _text(msp, text, x, y, layer="ROOM_TEXT"):
    t = msp.add_text(text, dxfattribs={"layer": layer, "height": 0.18})
    t.set_placement((x, y), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)


def _ellipse(msp, cx, cy, rx, ry):
    pts = [(cx + rx * math.cos(2 * math.pi * i / 48), cy + ry * math.sin(2 * math.pi * i / 48)) for i in range(48)]
    msp.add_lwpolyline(pts, close=False, dxfattribs={"layer": "FIXTURE"})


def build_dataset() -> None:
    d = ROOT / "data" / "swiss_dwellings"
    d.mkdir(parents=True)
    doc = ezdxf.new("R2013")
    msp = doc.modelspace()
    _rect(msp, "ROOM_OUTLINE", 0, 0, 4, 3)          # 12 m2 room
    _rect(msp, "ROOM_OUTLINE", 4, 0, 6, 2)          # 4 m2 bathroom
    _rect(msp, "ROOM_OUTLINE", 0, 3, 2, 4)          # 2 m2 balcony (outdoor)
    _rect(msp, "UNIT_OUTLINE", -0.2, -0.2, 6.2, 4.2)
    for text, x, y in [("Zimmer", 2, 1.7), ("1.01", 2, 1.4), ("Bad", 5, 1.2), ("1.02", 5, 0.9), ("Balkon", 1, 3.6), ("1.03", 1, 3.4), ("Whg. A", 2, 2.4)]:
        _text(msp, text, x, y)
    _rect(msp, "DOOR", 3.9, 0.5, 4.1, 1.4, five=True)
    _rect(msp, "WINDOW", 1, -0.1, 2.2, 0.1)
    _rect(msp, "WINDOW", 5.9, 0.6, 6.1, 1.4)
    _rect(msp, "FIXTURE", 4.2, 0.1, 4.6, 0.8)       # toilet: outline + bowl ellipse
    _ellipse(msp, 4.4, 0.5, 0.15, 0.22)
    _rect(msp, "FIXTURE", 5.2, 0.1, 5.8, 0.5)       # washbasin: outline + bowl ellipse + drain
    _ellipse(msp, 5.5, 0.3, 0.2, 0.12)
    msp.add_circle((5.5, 0.3), 0.02, dxfattribs={"layer": "FIXTURE"})
    doc.saveas(d / f"{STEM}.dxf")
    # the sheet: a PDF with the room numbers printed where the DXF has them (40 pt per metre), and its PNG at 2x
    import pymupdf

    pdf = pymupdf.open()
    page = pdf.new_page(width=420, height=300)
    for text, x, y in [("1.01", 2, 1.4), ("1.02", 5, 0.9), ("1.03", 1, 3.4)]:
        w = pymupdf.get_text_length(text, fontsize=9)
        page.insert_text((50 + 40 * x - w / 2, 250 - 40 * y + 3), text, fontsize=9)
    pdf.save(d / f"{STEM}.pdf")
    page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).save(d / f"{STEM}.png")
    path = f"data/swiss_dwellings/{STEM}.pdf"
    rows = [
        ("How many doors (interior doors and entrance doors together) are drawn on this floor plan?", 1, "count", "count"),
        ("How many windows are drawn on this floor plan?", 2, "count", "count"),
        ("How many rooms labelled 'Zimmer' are there on this floor?", 1, "count", "count"),
        ("What is the net floor area of room 1.01 ('Zimmer') in m²?", 12.0, "m2", "area"),
    ]
    with open(ROOT / "questions.jsonl", "w", encoding="utf-8") as fp:
        for q, a, unit, typ in rows:
            fp.write(json.dumps({"path": path, "question": q, "answer": a, "unit": unit, "type": typ}, ensure_ascii=False) + "\n")
    (d / "LICENSE.md").write_text("# test\n\nSynthetic floor generated for unit tests.\n")


build_dataset()
os.environ["STOA_DATASET"] = str(ROOT)
os.environ["STOA_NO_LLM"] = "1"  # tests never call a model
