"""Cross-sheet references (stoa_qa.xref) on a synthetic two-sheet set written into the test dataset.

    A-101  plan: a detail bubble "3" over "A-501", and a note "SILL FLASHING (SEE 2/A-501)"
    A-501  details: views titled "2  SILL DETAIL" and "3  WALL SECTION", each with a scale line
"""
import pymupdf
import pytest

from tests.conftest import ROOT

PROJECT = "data/opening_design/Test_Set"
PLAN = f"{PROJECT}/Sheets/A-101 - First Floor Plan.pdf"
DETAILS = f"{PROJECT}/Sheets/A-501 - Details.pdf"


def _sheet(path, number, texts):
    doc = pymupdf.open()
    page = doc.new_page(width=1224, height=792)
    for (x, y), text, size in texts:
        page.insert_text((x, y), text, fontsize=size)
    page.insert_text((1060, 760), number, fontsize=28)  # title block: the biggest text, bottom right
    out = ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)


@pytest.fixture(scope="module", autouse=True)
def drawing_set(tmp_path_factory):
    _sheet(PLAN, "A-101", [((300, 300), "3", 9), ((292, 312), "A-501", 9),
                           ((500, 420), "SILL FLASHING (SEE 2/A-501)", 8), ((100, 700), "FIRST FLOOR PLAN", 10)])
    _sheet(DETAILS, "A-501", [((100, 400), "2", 12), ((122, 400), "SILL DETAIL", 12), ((122, 416), 'SCALE: 3" = 1\'-0"', 9),
                              ((600, 400), "3", 12), ((622, 400), "WALL SECTION", 12), ((622, 416), 'SCALE: 1/2" = 1\'-0"', 9),
                              ((150, 250), "FLASHING", 8), ((650, 250), "STUD WALL", 8)])
    from stoa_qa import xref

    xref.CACHE = tmp_path_factory.mktemp("xref")  # keep the sheet index cache out of the repo
    xref.index.cache_clear()
    yield


def test_sheet_numbers_come_from_the_title_block():
    from stoa_qa import xref

    assert xref.own_number(PLAN) == "A-101" and xref.own_number(DETAILS) == "A-501"
    assert xref.find_sheet(PLAN, "A501")["path"] == DETAILS  # A501 and A-501 are the same sheet


def test_callouts_bubble_and_inline():
    from stoa_qa import xref

    found = {(c.kind, c.detail, c.sheet) for c in xref.callouts(PLAN)}
    assert ("bubble", "3", "A-501") in found
    assert ("inline", "2", "A-501") in found


def test_targets_are_the_titled_views():
    from stoa_qa import xref

    titles = {v["detail"]: v["title"] for v in xref.view_titles(DETAILS)}
    assert titles == {"2": "SILL DETAIL", "3": "WALL SECTION"}
    where = xref.locate(DETAILS, 0, "3")
    assert where["title"] == "WALL SECTION" and where["bbox"][3] >= 400  # the crop ends at the view's title


def test_reference_answers_and_side_by_side_pairs():
    from stoa_qa import xref

    a = xref.answer(PLAN, "Where is detail 3 on A-501?")
    assert a.value.startswith("Detail 3 on sheet A-501: WALL SECTION")
    assert a.references and a.references[0]["to"]["path"] == DETAILS and a.references[0]["from"]["box"]
    a = xref.answer(PLAN, "Which other sheets does this sheet refer to?")
    assert "A-501: 2 callout(s)" in a.value
    assert xref.answer(PLAN, "How many doors are on this plan?") is None
    assert xref.answer(PLAN, "Where is detail 9 on A-999?").value is None  # sheet not in the set: no guess
