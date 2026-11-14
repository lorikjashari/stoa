"""Refusals that explain themselves (stoa_qa.explain) and the pipeline's routing fixes."""
from tests.conftest import STEM

SWISS = f"data/swiss_dwellings/{STEM}.pdf"


def test_categories_from_what_the_engines_said():
    from stoa_qa.core import Answer
    from stoa_qa.explain import category

    assert category(Answer(None, intent="unanswerable", method="Not answerable from this drawing's data: no heights"), [], True) == "not_in_data"
    assert category(Answer(None, method="vision"), [{"note": "No north arrow is visible in the image", "outcome": "declined"}], True) == "not_in_data"
    assert category(Answer(None, method="document model unavailable: bad reply: unterminated JSON"), [], True) == "failed"
    assert category(Answer(None, method="cubicasa: question not understood"), [], False) == "needs_llm"
    assert category(Answer(None, method="cubicasa: question not understood"), [], True) == "not_understood"
    # the planner's decline contract
    assert category(Answer(None, intent="plan_count", method="doors where fire = 60 → count · no rows match"), [], True) == "not_in_data"
    assert category(Answer(None, intent="plan_sum", method="walls → sum(u_value) · 'u_value' is not recorded in this drawing"), [], True) == "not_in_data"
    assert category(Answer(None, intent="error", method="planner produced an invalid query (PlanError: unknown table 'x'): {}"), [], True) == "not_understood"
    assert category(Answer(None, intent="error", method="planner unavailable: FEATHERLESS_API_KEY is not set"), [], True) == "needs_llm"
    assert category(Answer(None, intent="error", method="planner unavailable: HTTP 503: busy"), [], True) == "failed"


def test_alternatives_use_this_drawings_own_rooms():
    from stoa_qa import api
    from stoa_qa.core import Answer
    from stoa_qa.explain import explain

    w = explain(api._drawing(SWISS), "How tall is the Zimmer?", Answer(None, intent="unanswerable", method="no heights"), [], True)
    assert w["headline"].startswith("This drawing doesn't contain")
    assert "2D CAD floor plan" in w["detail"] and 1 <= len(w["alternatives"]) <= 4
    assert any("1.01 Zimmer" in q or "Zimmer" in q for q in w["alternatives"])  # closest first, named as the drawing names it


def test_templates_only_get_questions_of_their_shape():
    from stoa_qa.pipeline import template_shaped

    assert template_shaped("How many doors are drawn on this floor plan?")
    assert template_shaped("What is the net floor area of room 1.01 in m²?")
    for q in ["How many windows face south?", "What materials are used in the walls?", "What is the scale of this drawing?",
              "How many doors lead outside?", "Is there an elevator?"]:
        assert not template_shaped(q), q


def test_yes_no_questions_read_as_yes_no():
    from stoa_qa.core import Answer
    from stoa_qa.pipeline import presented

    assert presented("Is there a balcony?", Answer(0, "count")) == "No — none found"
    assert presented("Are there bathtubs?", Answer(2, "count")) == "Yes — 2 found"
    assert presented("How many doors?", Answer(2, "count")) == "2"


def test_room_width_from_the_outline():
    from stoa_qa import graph

    a = graph.answer(SWISS, "How wide is room 1.01?")
    assert a.value.startswith("1.01 Zimmer: 3.00 m wide × 4.00 m long")
