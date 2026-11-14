from stoa_qa.cubicasa_refine import apply, component_count, floor_count, wet_rooms


def _room(cls, x0, y0, x1, y1):
    return {"class": cls, "poly": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def test_one_floor_is_a_single_group():
    rooms = [_room("Bed Room", 0, 0, 4, 3), _room("Kitchen", 4, 0, 7, 3), _room("Outdoor", 20, 0, 30, 10)]
    assert floor_count(rooms) == 1


def test_two_storeys_drawn_side_by_side():
    left = [_room("Bed Room", 0, 0, 5, 4), _room("Kitchen", 5, 0, 9, 4)]
    right = [_room("Bed Room", 14, 0, 19, 4), _room("Bath", 19, 0, 22, 4)]
    assert floor_count(left + right) == 2


def test_each_wet_room_is_one_fixture_and_a_small_one_is_a_wc():
    rooms = [_room("Bath", 0, 0, 40, 40), _room("Bath", 0, 0, 100, 120)]  # 1600 and 12000 at scale 1
    fixtures, wc, bathrooms = wet_rooms(rooms, scale=1.0)
    assert fixtures == 2 and wc == 1 and bathrooms == 1


def test_component_count_drops_specks():
    assert component_count([10, 40, 39, 200], 40) == 2


def test_apply_replaces_polygon_counts_and_adds_the_missing_questions():
    pred = {
        "scale": 1.0,
        "counts": {"doors": 3, "windows": 1, "toilets": 0, "sinks": 0, "baths": 0, "bedrooms": 1},
        "rooms": [_room("Bed Room", 0, 0, 4, 3), _room("Bath", 4, 0, 6, 2)],
    }
    seg = {"icons": {"Door": [50, 50, 5], "Window": [130], "Sink": [90, 10]}, "rooms": {"Bed Room": [8000, 100]}}
    got = apply(pred, seg)
    assert got["doors"] == 2 and got["windows"] == 1 and got["toilets"] == 0 and got["sinks"] == 1
    assert got["baths"] == 1 and got["floors"] == 1 and got["bedrooms"] == 1


# ---- annotation -> detector label rules (stoa_qa.cubicasa_rules)
from stoa_qa.cubicasa_rules import plan_boxes  # noqa: E402


def _obj(cls, b, t=""):
    return {"c": cls, "b": b, "t": t, "f": 0}


def _count(plan, cls):
    return sum(lab == cls for lab, _ in plan_boxes(plan))


def test_shower_inside_a_shower_cabin_counts_once():
    plan = {"objs": [_obj("FixedFurniture ShowerCab", [0, 0, 40, 40]), _obj("FixedFurniture Shower", [10, 30, 25, 40]),
                     _obj("FixedFurniture Bathtub", [100, 0, 180, 40])], "floors": []}
    assert _count(plan, "shower") + _count(plan, "bathtub") == 2


def test_wc_label_decides_wc_room_versus_bathroom():
    plan = {"objs": [_obj("Space Bath", [0, 0, 10, 10], "WC"), _obj("Space Bath", [20, 0, 40, 20], "KPH"),
                     _obj("Space Bath Shower", [50, 0, 60, 10], "PH"), _obj("Space UserDefined", [70, 0, 80, 10], "WC/PH")],
            "floors": []}
    assert _count(plan, "wc_room") == 2 and _count(plan, "bathroom") == 2


def test_sauna_is_the_stove_and_open_doorways_do_not_count():
    plan = {"objs": [_obj("FixedFurniture ElectricalAppliance SaunaStove", [0, 0, 5, 5]),
                     _obj("FixedFurniture SaunaBenchHigh", [5, 0, 20, 5]), _obj("Space Sauna", [0, 0, 20, 20]),
                     _obj("Door Swing Beside", [30, 0, 40, 10]), _obj("Door None", [50, 0, 60, 10]),
                     _obj("Space Outdoor Balcony Glazed", [0, 50, 30, 70], "LASITETTU PARVEKE"),
                     _obj("Space Outdoor Terrace", [40, 50, 70, 70], "TERASSI")],
            "floors": [{"name": "Floorplan", "b": [0, 0, 70, 70]}]}
    assert _count(plan, "sauna") == 1 and _count(plan, "door") == 1
    assert _count(plan, "balcony") == 1 and _count(plan, "floor") == 1


def test_select_boxes_threshold_and_cross_class_duplicates():
    from stoa_qa.cubicasa_rules import CLASSES, select_boxes

    sh, tub, door = CLASSES.index("shower"), CLASSES.index("bathtub"), CLASSES.index("door")
    boxes = [[sh, 0.9, 0, 0, 10, 10], [tub, 0.6, 1, 1, 10, 10], [tub, 0.8, 50, 50, 70, 60],
             [sh, 0.2, 100, 100, 110, 110], [door, 0.99, 0, 0, 10, 10]]
    assert len(select_boxes(boxes, "baths", 0.5)) == 3          # no suppression: shower and tub on one fixture both count
    assert len(select_boxes(boxes, "baths", 0.5, iou=0.5)) == 2  # the overlapping lower-scoring tub is dropped
    assert len(select_boxes(boxes, "baths", 0.1, iou=0.5)) == 3  # low threshold admits the faint shower
    assert len(select_boxes(boxes, "doors", 0.5)) == 1


def test_ensemble_fuses_shared_boxes_and_discounts_lone_ones():
    from stoa_qa.cubicasa_ensemble import fuse
    from stoa_qa.cubicasa_rules import CLASSES

    w, d = CLASSES.index("window"), CLASSES.index("door")
    a = [[w, 0.8, 0, 0, 10, 4], [w, 0.6, 50, 0, 60, 4]]
    b = [[w, 0.6, 1, 0, 10, 4], [d, 0.9, 0, 0, 10, 4]]
    out = sorted(fuse(a, b, lone=0.5), key=lambda r: -r[1])
    assert out[0][:2] == [w, 0.7]                              # both models saw it: averaged score
    assert [d, 0.45] == out[1][:2] and [w, 0.3] == out[2][:2]  # seen once: half the score; classes never mix
    assert len(out) == 3
