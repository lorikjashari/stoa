"""Floor-plan reconstruction (stoa_qa.floorplan) and the 3D scenes built from it (stoa_qa.plan3d), offline.

The CubiCasa5K network and the YOLO detector are both mocked: no torch, no weights, no network."""
import importlib.util
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Polygon

from stoa_qa import cubicasa, floorplan as FP, plan3d
from stoa_qa.cubicasa_rules import CLASSES
from tests.test_live_detect import _png, live  # noqa: F401  (the mocked-detector fixture)

C = {c: CLASSES.index(c) for c in CLASSES}
# the reconstruction needs the raster extra (OpenCV, scikit-image); the base install and CI skip those tests
raster = pytest.mark.skipif(not all(importlib.util.find_spec(m) for m in ("cv2", "skimage")),
                            reason="needs the raster extra: uv sync --extra raster")
H, W = 300, 400
BED, LIVING = FP.ROOM_CLASSES.index("Bed Room"), FP.ROOM_CLASSES.index("Living Room")


def synthetic_maps(factor: float = 1.0) -> FP.Maps:
    """What the network 'sees' on a two-room plan: outer walls, a partition at x=200 with a door gap at y 120-165."""
    wall = np.zeros((H, W), np.float32)
    wall[40:48, 40:360] = wall[252:260, 40:360] = 1           # top / bottom walls (8 px thick)
    wall[40:260, 40:48] = wall[40:260, 352:360] = 1           # left / right walls
    wall[48:120, 196:204] = wall[165:252, 196:204] = 1        # partition with a door gap
    room = np.zeros((H, W), np.uint8)
    room[48:252, 48:196], room[48:252, 204:352] = BED, LIVING
    room[wall > 0.5] = 2
    z = np.zeros((H, W), np.float32)
    return FP.Maps(factor, wall, room, np.ones((H, W), np.float32), z, z.copy(), {"vec_walls": []})


DOOR_BOX = (0, "door", 200, 120, 245, 165)      # swing arc to the right of the partition
WIN_BOX = (1, "window", 100, 38, 160, 50)       # in the top wall
FURNITURE = (2, "door", 280, 180, 300, 200)     # a "door" in the middle of the living room


@raster
def test_reconstruct_cuts_openings_into_walls_and_separates_rooms():
    plan = FP.reconstruct(np.full((H, W, 3), 255, np.uint8), [DOOR_BOX, WIN_BOX, FURNITURE], maps=synthetic_maps())
    assert 6 <= plan.thick <= 10
    door, win, furn = plan.openings
    assert door.on_wall and not door.horizontal and 35 <= door.width <= 55          # in the vertical partition
    assert win.on_wall and win.horizontal                                           # in the top wall
    assert not furn.on_wall                                                         # furniture: no wall around it
    assert sorted(r["cls"] for r in plan.rooms) == ["Bed Room", "Living Room"]      # the door gap does not merge them
    walls = [Polygon(o, hs).buffer(0) for o, hs in plan.walls]
    probe = Polygon([(198, 130), (202, 130), (202, 150), (198, 150)])
    assert not any(w.intersects(probe) for w in walls)                              # the doorway is open in 3D


@raster
def test_clean_background_clears_margins_but_keeps_walls_that_reach_the_edge():
    img = np.full((200, 300, 3), 255, np.uint8)
    img[:30] = 0                      # a black page margin
    img[30:, :20] = 128               # grey site fill down the left side
    img[100:106, 20:300] = 40         # a wall running into the right edge
    out = FP.clean_background(img)
    assert (out[:30] == 255).all() and (out[30:, :20] == 255).all()
    assert (out[100:106, 150:300] == 40).all()


@raster
def test_sheet_frame_is_not_a_wall():
    m = np.zeros((300, 400), bool)
    m[2:6, :] = m[-6:-2, :] = m[:, 2:6] = m[:, -6:-2] = True  # the drawing's border frame
    m[100:108, 100:300] = True                                  # a real wall
    out = FP._drop_frame(m, 0.04)
    assert not out[:10].any() and out[100:108, 100:300].all()


def test_small_plans_are_enlarged_big_ones_never_shrunk():
    assert FP.choose_factor(np.full((500, 600, 3), 255, np.uint8)) == pytest.approx(1250 / 600)
    assert FP.choose_factor(np.full((300, 380, 3), 255, np.uint8)) == 3.0      # capped
    assert FP.choose_factor(np.full((2400, 2000, 3), 255, np.uint8)) == pytest.approx(1800 / 2400)  # big: shrunk for CPU time


@raster
def test_confidence_scale_is_where_the_network_is_surest_of_walls(monkeypatch):
    def fake(rgb, factor, vector=True):
        m = synthetic_maps(factor)
        m.wall[m.wall > 0.5] = {1.0: 0.6, 1.5: 0.7, 2.2: 0.95, 3.0: 0.8}[factor]
        return m
    monkeypatch.setattr(FP, "network_maps", fake)
    P = FP.Params(scale_mode="confidence")
    assert FP.choose_factor(np.full((500, 600, 3), 255, np.uint8), P) == 2.2
    assert FP.choose_factor(np.full((2400, 2000, 3), 255, np.uint8), P) == 1.0  # big plans are never enlarged past MAX_SIDE


@pytest.fixture
def plan_img(tmp_path):
    f = tmp_path / "plan.png"
    f.write_bytes(_png(W, H))
    return f


def _boxes():
    return [[C["door"], 0.9, 200, 120, 245, 165], [C["window"], 0.9, 100, 38, 160, 50], [C["door"], 0.9, 280, 180, 300, 200],
            [C["toilet"], 0.8, 60, 60, 70, 75], [C["bedroom"], 0.95, 50, 50, 194, 250]]


@raster
def test_scene_elements_carry_the_evidence_ids(plan_img, monkeypatch, tmp_path):
    monkeypatch.setattr(plan3d, "PLANS", tmp_path / "plans")
    monkeypatch.setattr(FP, "network_maps", lambda rgb, factor, vector=True: synthetic_maps(factor))
    monkeypatch.setattr(FP, "choose_factor", lambda rgb, params=None: 1.0)
    cfg, boxes = cubicasa.picks(), _boxes()
    sc = plan3d.build(plan_img, "uploads/cubicasa5k/x.png", boxes, cfg)
    assert sc["source"]["engine"] == "cubicasa5k-model+yolo"
    assert [d["id"] for d in sc["doors"]] == ["yolo:0"]            # the furniture "door" gets no opening
    assert [w["id"] for w in sc["windows"]] == ["yolo:1"] and [f["id"] for f in sc["fixtures"]] == ["yolo:3"]
    door = sc["doors"][0]
    leaf_x = [p[0] for p in door["leaf"]]
    assert min(leaf_x) >= min(p[0] for p in door["lintel"]) - 1e-6  # the leaf opens towards the swing arc (to the right)
    rooms = {r["id"]: r for r in sc["rooms"]}
    assert rooms["yolo:4"]["kind"] == "bedroom" and any(r["label"] == "Living Room" for r in sc["rooms"])
    shapes = [Polygon(r["poly"], r["holes"]).buffer(0) for r in sc["rooms"]]
    assert all(a.intersection(b).area < 1e-6 for i, a in enumerate(shapes) for b in shapes[i + 1:])  # floors never overlap
    pred = {"counts": {}, "boxes": boxes}
    for key, group in (("windows", "windows"), ("toilets", "fixtures"), ("bedrooms", "rooms")):
        ids = {e.viewer_id for e in cubicasa._evidence("yolo", key, pred, cfg)}
        assert ids and ids <= {el["id"] for el in sc[group]}


@raster
def test_upload_gets_a_3d_view_whose_marks_are_the_answers(live, monkeypatch, tmp_path):  # noqa: F811
    from stoa_qa import api

    monkeypatch.setattr(plan3d, "available", lambda: True)
    monkeypatch.setattr(FP, "network_maps", lambda rgb, factor, vector=True: synthetic_maps(factor))
    monkeypatch.setattr(FP, "choose_factor", lambda rgb, params=None: 1.0)
    monkeypatch.setattr(api, "SCENES", tmp_path / "scenes")
    c = TestClient(api.app)
    d = c.post("/api/upload", content=_png(W, H), headers={"content-type": "image/png"}).json()
    assert d["has_3d"] and d["model_kind"] == "raster" and d["scene"].endswith("-y")
    r = c.post("/api/scene", json={"id": d["id"], "question": ""})
    assert r.status_code == 200 and r.json()["url"].endswith("#" + d["scene"])
    sc = json.loads((tmp_path / "scenes" / f"{d['scene']}.json").read_text(encoding="utf-8"))
    assert sc["source"]["file"] == d["id"]                           # the viewer shows the upload as the original drawing
    a = c.post("/api/ask", json={"id": d["id"], "question": "How many bedrooms are there?"}).json()
    assert a["value"] == 1 and set(a["ids"]) <= {x["id"] for x in sc["rooms"]}
    assert c.post("/ask", json={"path": d["id"], "question": "How many bedrooms are there?"}).json()["value"] == 1


@raster
def test_doors_in_no_wall_are_not_counted_on_new_drawings(live, monkeypatch):  # noqa: F811
    """The mocked detector's three fused doors sit at y 10-20; a wall runs under the first two only."""
    from stoa_qa import uploads

    def maps(rgb, factor, vector=True):
        m = synthetic_maps(factor)
        m.wall[12:18, 0:150] = 1.0
        return m
    monkeypatch.setattr(plan3d, "available", lambda: True)
    monkeypatch.setattr(FP, "network_maps", maps)
    monkeypatch.setattr(FP, "choose_factor", lambda rgb, params=None: 1.0)
    rel = uploads.save(_png(W, H))
    a = cubicasa.answer(rel, "How many doors are there?")
    assert a.value == 2 and len(a.evidence) == 2
    assert "1 door/window detection not set in any wall was left out" in a.method


def test_no_3d_view_without_the_network(live, monkeypatch):  # noqa: F811
    from stoa_qa import api

    monkeypatch.setattr(plan3d, "available", lambda: False)
    c = TestClient(api.app)
    d = c.post("/api/upload", content=_png(), headers={"content-type": "image/png"}).json()
    assert not d["has_3d"] and d["scene"] is None
    assert c.post("/api/scene", json={"id": d["id"], "question": ""}).status_code == 404
