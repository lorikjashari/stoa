from fastapi.testclient import TestClient

from tests.conftest import STEM

ID = f"data/swiss_dwellings/{STEM}.pdf"


def client():
    from stoa_qa.api import app

    return TestClient(app)


def test_catalog_and_drawing():
    c = client()
    assert c.get("/api/health").json()["ok"] is True
    listing = c.get("/api/drawings", params={"only_questions": True}).json()
    assert listing["total"] == 1 and listing["items"][0]["id"] == ID
    d = c.get("/api/drawing", params={"id": ID}).json()
    assert d["has_3d"] and d["question_count"] == 4 and d["questions"][0]["lang"] == "en"


def test_ask_returns_answer_evidence_and_ground_truth_check():
    r = client().post("/api/ask", json={"id": ID, "question": "How many windows are drawn on this floor plan?"}).json()
    assert r["value"] == 2 and r["expected"] == 2 and r["correct"] is True
    assert r["engine"] == {"kind": "geometry", "note": "Computed from CAD geometry (DXF)", "llm_tokens": 0, "cost_usd": 0.0, "models": []}
    assert r["ids"] and all(i.startswith("window:") for i in r["ids"])


def test_free_form_question_without_llm_is_refused_not_guessed():
    r = client().post("/api/ask", json={"id": ID, "question": "Which room is the largest?"}).json()
    assert r["value"] is None  # refused, never guessed
    why = r["unanswered"]  # and the refusal says why and what to ask instead (stoa_qa.explain)
    assert why["category"] == "needs_llm" and r["text"] == why["headline"]
    assert why["alternatives"] and all(q != "Which room is the largest?" for q in why["alternatives"])


def test_only_dataset_files_are_served():
    c = client()
    assert c.get("/api/file", params={"path": "../.env"}).status_code == 404
    assert c.get("/api/file", params={"path": f"data/swiss_dwellings/{STEM}.dxf"}).status_code == 200
    assert c.get("/.env").status_code == 404
    assert c.post("/api/ask", json={"id": "data/nope.pdf", "question": "x"}).status_code == 404


def test_evidence_is_mapped_onto_the_sheet_pixels():
    from stoa_qa import sheets

    s, tx, ty = sheets.swiss_transform(STEM)          # 40 pt per metre, PNG rendered at 2x -> 80 px per metre
    assert abs(s - 80) < 1.5 and abs(tx - 100) < 4 and abs(ty - 500) < 6
    r = client().post("/api/ask", json={"id": ID, "question": "What is the net floor area of room 1.01 ('Zimmer') in m²?"}).json()
    item = r["overlay"]["items"][0]
    assert r["overlay"]["space"] == "px" and item["kind"] == "poly"
    x0, y0, x1, y1 = item["bbox"]                      # room 1.01 spans (0,0)-(4,3) m
    assert abs(x0 - 100) < 5 and abs(x1 - 420) < 5 and abs(y1 - 500) < 7 and abs(y0 - 260) < 7
