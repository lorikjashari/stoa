"""blueprint3d/archive_plan.py: the floor-plan check for archive scans (no network, no dataset needed)."""
import os
import sys

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("PIL")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "blueprint3d"))
import archive_plan as A  # noqa: E402

PLAN = {"wall_frac": 0.09, "interior_frac": 0.4, "rooms_big": 8, "rooms_closed": 6, "closed_frac": 0.3,
        "band_frac": 0.0, "room_ink": 0.02, "ink_frac": 0.05, "wall_px": 6.0, "door_frac": 0.002, "window_frac": 0.01}


def view(box, parent=-1, children=(), labels=(), **f):
    return {"box": box, "parent": parent, "children": list(children), "labels": list(labels), "features": {**PLAN, **f}}


def test_plan_accepted_and_reason_kept():
    views = [view((0, 0, 1000, 800))]
    best, n = A.judge(views)
    assert best == 0 and n == 1 and views[0]["verdict"] == "plan"


def test_printed_section_title_vetoes_a_plan_looking_drawing():
    views = [view((0, 0, 1000, 800), labels=["SECTION B-B"])]
    assert A.judge(views)[0] is None
    assert views[0]["verdict"] == "other" and "SECTION B-B" in views[0]["why"]
    assert "SECTION B-B" in A._reason(views)


@pytest.mark.parametrize("title", ["LEIKKAUS A-1", "Fasad mot väster", "WEST ELEVATION", "Schnitt A-B", "Coupe AB"])
def test_titles_in_dataset_languages(title):
    views = [view((0, 0, 1000, 800), labels=[title])]
    assert A.judge(views)[0] is None


def test_plan_title_kept_and_long_notes_ignored():
    views = [view((0, 0, 1000, 800), labels=["FIRST FLOOR PLAN", "SEE SECTION A ON SHEET 4 FOR THE STAIR DETAILS"])]
    best, _ = A.judge(views)
    assert best == 0 and views[0]["label"][0] == "plan"


def test_height_levels_mean_section_or_elevation():
    views = [view((0, 0, 1000, 800), labels=["+12,510", "+7,780", "+2,600"])]
    assert A.judge(views)[0] is None


def test_elevation_rooms_full_of_linework_rejected():
    views = [view((0, 0, 1000, 800), room_ink=0.12)]
    assert A.judge(views)[0] is None and views[0]["verdict"] == "other"


def test_strip_is_not_a_plan():
    views = [view((0, 0, 3000, 600))]
    assert A.judge(views)[0] is None


def test_most_specific_plan_wins_and_other_drawing_blocks_parent():
    # sheet = plan (left) + elevation (right): the sheet view is blocked, the plan view is chosen
    views = [view((0, 0, 2000, 1000), children=[1, 2]),
             view((0, 0, 1000, 1000), parent=0),
             view((1000, 0, 2000, 1000), parent=0, labels=["SOUTH ELEVATION"])]
    best, n = A.judge(views)
    assert best == 1 and n == 1


def test_multi_view_sheet_without_isolated_plan_declined():
    # the whole sheet looks like a plan, but splits into several unlabelled drawings that are not plans
    weak = dict(rooms_closed=0, closed_frac=0.0)
    views = [view((0, 0, 2000, 2000), children=[1, 2]),
             view((0, 0, 2000, 900), parent=0, **weak),
             view((0, 1000, 2000, 2000), parent=0, **weak)]
    assert A.judge(views)[0] is None


def test_plan_fragments_do_not_block():
    # XY-cut splits one plan into strips at a courtyard: the plan is still accepted
    weak = dict(rooms_closed=1, closed_frac=0.05)
    views = [view((0, 0, 1200, 1000), children=[1, 2]),
             view((0, 0, 1200, 300), parent=0, **weak),
             view((0, 600, 1200, 1000), parent=0, **weak)]
    assert A.judge(views)[0] == 0


def test_find_views_splits_drawings_at_whitespace():
    sheet = np.full((1200, 1600, 3), 245, np.uint8)
    # two drawings made of rooms (walls shorter than the frame-rule length), 200 px of paper between them
    for x0, y0, x1, y1 in ((100, 100, 700, 1000), (900, 200, 1500, 900)):
        for x in range(x0, x1, 150):
            for y in range(y0, y1, 150):  # rooms 30 px apart, joined by a short corridor wall
                cv2.rectangle(sheet, (x, y), (min(x + 120, x1), min(y + 120, y1)), (20, 20, 20), 6)
                cv2.line(sheet, (x + 60, y + 120), (x + 60, min(y + 150, y1)), (20, 20, 20), 6)
                cv2.line(sheet, (x + 120, y + 60), (min(x + 150, x1), y + 60), (20, 20, 20), 6)
        cv2.rectangle(sheet, (x0 - 10, y0 - 10), (x1 + 10, y1 + 10), (20, 20, 20), 8)  # long outer walls
    views = A.find_views(sheet)
    boxes = [v["box"] for v in views]
    # each drawing on its own, outer walls included (they are long enough to be taken for frame rules)
    assert any(b[0] <= 90 and b[1] <= 90 and 710 <= b[2] < 800 and b[3] >= 1010 for b in boxes)
    assert any(b[0] > 800 and b[0] <= 890 and b[2] >= 1510 for b in boxes)
    assert views[0]["parent"] == -1 and len(views[0]["children"]) == 2


def test_blueprint_negative_is_inverted(tmp_path):
    from PIL import Image
    img = np.full((400, 400, 3), (40, 60, 160), np.uint8)
    cv2.rectangle(img, (50, 50), (350, 350), (230, 230, 230), 6)
    p = tmp_path / "blue.jpg"
    Image.fromarray(img).save(p)
    rgb = A.load_rgb(str(p))
    assert np.median(rgb) > 128  # light paper, dark lines
