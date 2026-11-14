import pytest

from tests.conftest import STEM

PATH = f"data/swiss_dwellings/{STEM}.pdf"


def ask(q):
    from stoa_qa.router import answer

    return answer(PATH, q)


@pytest.mark.parametrize("question, value", [
    ("How many doors (interior doors and entrance doors together) are drawn on this floor plan?", 1),
    ("Wie viele Türen (Innentüren und Eingangstüren zusammen) sind auf diesem Grundriss eingezeichnet?", 1),
    ("Combien de fenêtres sont dessinées sur ce plan ?", 2),
    ("Quante finestre sono disegnate in questa pianta?", 2),
    ("How many toilets (WC) are drawn on this floor plan?", 1),
    ("How many washbasins are drawn on this floor plan?", 1),
    ("How many bathtubs are drawn on this floor plan?", 0),
    ("How many apartments (residential units) are shown on this floor plan?", 1),
    ("How many rooms labelled 'Zimmer' are there on this floor?", 1),
    ("Wie viele Räume mit der Bezeichnung «Bad» gibt es in diesem Geschoss?", 1),
    ("Combien de pièces portant la désignation « Zimmer » y a-t-il à cet étage ?", 1),
])
def test_counts_in_four_languages(question, value):
    a = ask(question)
    assert a.value == value, a.method
    assert len([e for e in a.evidence if e.coords]) == value  # one located evidence item per counted thing


@pytest.mark.parametrize("question, value", [
    ("What is the net floor area of room 1.01 ('Zimmer') in m²?", 12.0),
    ("Wie gross ist die Nettofläche von Raum 1.02 («Bad») in m²?", 4.0),
    ("What is the total net area in m² of all rooms labelled 'Zimmer' on this floor?", 12.0),
    ("What is the total net floor area in m² of all numbered indoor rooms on this floor (excluding balconies, loggias, terraces and other outdoor spaces, shafts and voids)?", 16.0),
    ("What is the net floor area of apartment A in m², summing all its rooms except balconies, loggias, terraces and other outdoor spaces?", 16.0),
])
def test_areas_from_geometry(question, value):
    a = ask(question)
    assert a.value == pytest.approx(value, abs=0.01), a.method
    assert a.unit == "m2" and a.evidence


def test_unknown_question_is_not_guessed():
    a = ask("Who designed this building?")
    assert a.value is None


def test_language_parser():
    from stoa_qa.lang import is_model_scope, objects_in, quantity_of

    assert quantity_of("Wat is het totale netto volume van alle vloeren op 'X', in m³?") == "volume"
    assert quantity_of("Hvad er den samlede længde af alle vægge på 'X' i meter?") == "length"
    assert quantity_of("¿Cuál es el número de puertas en el plano de la planta «X»?") == "count"
    assert "door" in objects_in("Ile drzwi pokazano na rzucie kondygnacji „1”?")
    assert "window" in objects_in("Mitu akent on mudelis „X“?") and is_model_scope("Mitu akent on mudelis „X“?")
    assert "column" in objects_in("Care este numărul de stâlpi de pe planul nivelului „Level 1”?")


def test_planner_executes_queries_exactly():
    from stoa_qa import planner, swiss

    tables = swiss.facts(STEM)
    a = planner.execute({"table": "rooms", "where": [{"col": "outdoor", "op": "=", "val": False}], "agg": "top", "col": "area_m2", "n": 1}, tables)
    assert "1.01" in a.value and a.evidence[0].viewer_id == "room:1.01"
    assert planner.execute({"table": "rooms", "where": [{"col": "area_m2", "op": ">", "val": 3}], "agg": "count"}, tables).value == 2
    assert planner.execute({"table": "rooms", "where": [{"col": "toilets", "op": ">", "val": 0}], "agg": "sum", "col": "area_m2"}, tables).value == 4.0
    assert planner.execute({"table": "apartments", "where": [], "agg": "max", "col": "area_m2"}, tables).value == 16.0
    multi = planner.execute_many([{"title": "Rooms", "table": "rooms", "where": [], "agg": "count"},
                                  {"title": "Doors", "table": "doors", "where": [], "agg": "count"}], tables)
    assert multi.value == "Rooms: 3\nDoors: 1"
    with pytest.raises(planner.PlanError):
        planner.execute({"table": "rooms", "where": [{"col": "nope", "op": "=", "val": 1}], "agg": "count"}, tables)
    assert planner.execute({"unanswerable": "no architect in the data"}, tables).value is None


def test_scoring_rules():
    from stoa_qa.pipeline import is_correct

    assert is_correct(23, 23, "count") is True and is_correct(23, 22, "count") is False
    assert is_correct(100.0, 101.9, "area") is True and is_correct(100.0, 102.1, "area") is False
    assert is_correct("free text", "x", "text") is None


def test_sheet_rows_are_rebuilt_from_pdf_words(tmp_path, monkeypatch):
    import pymupdf

    from stoa_qa import docqa, sheets

    pdf = tmp_path / "schedule.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=300)
    for i, row in enumerate([("101.A", "OFFICE", "A1"), ("102.A", "LOBBY", "B2"), ("103.A", "STORE", "A1")]):
        for j, cell in enumerate(row):
            page.insert_text((40 + 90 * j, 60 + 18 * i), cell, fontsize=9)
    page.insert_text((470, 60), "GENERAL NOTE", fontsize=9)  # same height, far away: must not join the first row
    doc.save(pdf)
    monkeypatch.setattr(sheets, "safe_path", lambda rel: pdf)
    docqa.layout.cache_clear()
    rows, w, h = docqa.layout("schedule.pdf")
    texts = [r["text"] for r in rows]
    assert "101.A | OFFICE | A1" in texts and "GENERAL NOTE" in texts
    assert (w, h) == (600, 300)
