"""Scoped questions ("windows in the bathroom", "doors of apartment A") are counted inside that scope or
declined, never answered with the floor-wide number. Synthetic floor from conftest: Zimmer 1.01 (12 m²),
Bad 1.02 (4 m², toilet + washbasin), Balkon 1.03; apartment A; one door Zimmer↔Bad; one window each in Zimmer and Bad."""
import pytest

from tests.conftest import STEM

PATH = f"data/swiss_dwellings/{STEM}.pdf"


def ask(q):
    from stoa_qa.router import answer

    return answer(PATH, q)


@pytest.mark.parametrize("question, value", [
    ("How many windows are in the bathroom?", 1),
    ("Wie viele Fenster hat das Bad?", 1),
    ("How many windows does room 1.01 have?", 1),
    ("How many toilets are in the bathroom?", 1),
    ("How many doors does the bathroom have?", 1),
    ("How many windows are in apartment A?", 2),
    ("How many doors does apartment A have?", 1),
    ("How many rooms labelled 'Zimmer' are in apartment A?", 1),
])
def test_counts_inside_the_named_scope(question, value):
    a = ask(question)
    assert a.value == value, a.method
    assert a.confidence >= 0.85


@pytest.mark.parametrize("question", [
    "How many bathrooms does apartment A have?",   # used to answer 1 = the number of apartments
    "How many windows are in the kitchen?",        # no kitchen on this floor: used to answer 2 = every window
    "How many apartments have a balcony?",
])
def test_declines_what_it_cannot_scope(question):
    a = ask(question)
    assert a.value is None, f"{a.value}: {a.method}"


@pytest.mark.parametrize("question, value", [
    ("What is the net floor area of apartment A?", 16.0),   # no unit written: still an area question
    ("What is the area of the bathroom?", 4.0),
    ("What is the net floor area of apartment A in m², summing all its rooms except balconies, loggias, terraces "
     "and other outdoor spaces?", 16.0),                     # "except balconies" is an exclusion, not the scope
])
def test_areas_of_the_named_scope(question, value):
    a = ask(question)
    assert a.value == pytest.approx(value, rel=0.02), a.method


@pytest.mark.parametrize("question, value, intent", [
    ("What is the total living area?", 16.0, "living_area"),     # the apartments, not a room labelled "living"
    ("Wie gross ist die Wohnfläche?", 16.0, "living_area"),
    ("What is the total area of all bathrooms?", 4.0, "room_area"),  # one bathroom here: its area
    ("What is the total floor area?", 16.0, "floor_area"),
])
def test_area_phrases_are_quantities_not_rooms(question, value, intent):
    a = ask(question)
    assert a.value == pytest.approx(value, rel=0.02) and a.intent == intent, a.method


def test_area_of_a_room_the_floor_does_not_have_is_declined():
    a = ask("What is the area of the living room?")
    assert a.value is None, a.method


def test_auditor_fails_a_floor_wide_answer_to_a_scoped_question():
    from stoa_qa.audit import scope_check
    from stoa_qa.core import Answer

    wide = Answer(28, "count", "windows", 0.97, "28 window symbols")
    assert [c.status for c in scope_check("How many windows are in the kitchen?", wide)] == ["fail"]
    assert [c.status for c in scope_check("How many bathrooms does apartment B have?", Answer(4, "count", "apartments", 0.97, ""))] == ["fail"]
    assert scope_check("How many windows are drawn on this floor?", wide) == []
    assert scope_check("How many toilets (WC) are drawn on this floor?", Answer(4, "count", "toilets", 0.95, "")) == []
    # "vasche da bagno" are bathtubs; the bagno in them is not a bathroom
    assert scope_check("Quante vasche da bagno sono disegnate in questa pianta?", Answer(2, "count", "bathtubs", 0.95, "")) == []


def test_italian_bathtubs_are_counted_on_the_whole_floor():
    a = ask("Quante vasche da bagno sono disegnate in questa pianta?")
    assert a.intent == "bathtubs" and a.value == 0, a.method


def test_question_understanding():
    from stoa_qa.lang import objects_in, quantity_of

    assert quantity_of("What is the net floor area of apartment B?") == "area"
    assert quantity_of("Wie gross ist die Fläche der Küche?") == "area"
    assert quantity_of("How many rooms labelled 'Fläche' are there?") == "count"
    assert "stair" not in objects_in("What is the scale of this drawing?")
    assert "stair" in objects_in("How many stairs are there?")
