"""Live YOLO detection on plans without precomputed boxes (stoa_qa.cubicasa_live) and the upload path.

The detector is mocked: no ultralytics import, no weights, no network."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from stoa_qa import cubicasa, cubicasa_live, uploads
from stoa_qa.cubicasa_ensemble import fuse
from stoa_qa.cubicasa_rules import CLASSES

DOOR, WINDOW, BED = CLASSES.index("door"), CLASSES.index("window"), CLASSES.index("bedroom")
# what each (mocked) model "sees": three doors both find, one door only the big model finds, a bedroom only the small one
BOXES = {
    "m13-1280": [[DOOR, 0.9, 10, 10, 30, 20], [DOOR, 0.85, 100, 10, 120, 20], [DOOR, 0.8, 200, 10, 220, 20],
                 [DOOR, 0.7, 300, 10, 320, 20]],
    "local-y11s-13c": [[DOOR, 0.9, 11, 10, 30, 21], [DOOR, 0.85, 100, 11, 121, 20], [DOOR, 0.8, 201, 10, 220, 20],
                       [BED, 0.95, 50, 50, 150, 150]],
}


class _T(list):
    def tolist(self):
        return list(self)


class _Result:
    def __init__(self, rows):
        self.boxes = type("Boxes", (), {"cls": _T(r[0] for r in rows), "conf": _T(r[1] for r in rows),
                                        "xyxy": _T(r[2:6] for r in rows)})()


class FakeYOLO:
    def __init__(self, run, calls):
        self.run, self.calls = run, calls

    def predict(self, source, imgsz, conf, verbose):
        self.calls.append((self.run, imgsz, conf))
        return [_Result(BOXES[self.run])]


@pytest.fixture
def live(tmp_path, monkeypatch):
    """Both members of the picked fused run 'installed', uploads and caches in a temp folder."""
    weights = tmp_path / "weights"
    weights.mkdir()
    for run in BOXES:
        (weights / f"{run}.pt").write_bytes(b"not a real checkpoint")
    calls: list = []
    monkeypatch.setenv("STOA_YOLO_WEIGHTS", str(weights))
    monkeypatch.setenv("STOA_UPLOADS", str(tmp_path / "uploads"))
    monkeypatch.delenv("STOA_NO_LIVE_YOLO", raising=False)
    monkeypatch.setattr(cubicasa_live, "CACHE", tmp_path / "live")
    monkeypatch.setattr(cubicasa_live, "has_ultralytics", lambda: True)
    monkeypatch.setattr(cubicasa_live, "_load", lambda run: FakeYOLO(run, calls))
    from stoa_qa import plan3d  # the wall check needs the CubiCasa5K network: off unless a test mocks it in

    monkeypatch.setattr(plan3d, "available", lambda: False)
    monkeypatch.setattr(plan3d, "PLANS", tmp_path / "plans")
    return {"weights": weights, "calls": calls, "tmp": tmp_path}


def _png(w=400, h=300) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def _plan(tmp_path):
    f = tmp_path / "plan.png"
    f.write_bytes(_png())
    return f


def test_picked_run_is_the_fused_pair():
    assert cubicasa.picks()["yolo_run"] == "ens-m13-s13"
    assert cubicasa_live.members("ens-m13-s13") == ("m13-1280", "local-y11s-13c")


def test_both_models_run_at_their_own_size_and_are_fused(live):
    out = cubicasa_live.boxes("ens-m13-s13", _plan(live["tmp"]))
    assert out["fused"] and out["complete"] and out["runs"] == ["m13-1280", "local-y11s-13c"]
    assert sorted(live["calls"]) == [("local-y11s-13c", 1024, 0.05), ("m13-1280", 1280, 0.05)]
    assert out["boxes"] == fuse(BOXES["m13-1280"], BOXES["local-y11s-13c"])


def test_boxes_are_cached_per_image_and_run(live):
    plan = _plan(live["tmp"])
    first = cubicasa_live.boxes("ens-m13-s13", plan)
    second = cubicasa_live.boxes("ens-m13-s13", plan)
    assert first == second and len(live["calls"]) == 2  # the second call read the cache
    assert (live["tmp"] / "live" / first["hash"] / "m13-1280.json").is_file()


def test_one_model_alone_is_used_unfused(live):
    (live["weights"] / "local-y11s-13c.pt").unlink()
    out = cubicasa_live.boxes("ens-m13-s13", _plan(live["tmp"]))
    assert out["runs"] == ["m13-1280"] and not out["fused"] and not out["complete"]
    assert out["boxes"] == BOXES["m13-1280"]


def test_nothing_runs_without_ultralytics_or_weights(live, monkeypatch):
    plan = _plan(live["tmp"])
    monkeypatch.setattr(cubicasa_live, "has_ultralytics", lambda: False)
    assert cubicasa_live.boxes("ens-m13-s13", plan) is None
    monkeypatch.setattr(cubicasa_live, "has_ultralytics", lambda: True)
    for f in live["weights"].iterdir():
        f.unlink()
    assert cubicasa_live.boxes("ens-m13-s13", plan) is None
    assert "yolo" not in cubicasa.predictions("plan", plan)
    assert not live["calls"]


def test_counts_use_the_val_thresholds_and_each_box_is_evidence(live):
    rel = uploads.save(_png())
    a = cubicasa.answer(rel, "How many doors are there?")
    # three doors both models found (score ~0.85 >= 0.4); the one only m13 found keeps half its score (0.35) and drops out
    assert a.value == 3 and a.unit == "count" and a.intent == "doors"
    assert len(a.evidence) == 3 and all(e.kind == "box" and e.ref.startswith("yolo-") and e.coords for e in a.evidence)
    assert "Detected on this image just now by YOLO11 (m13-1280 + local-y11s-13c, fused)" in a.method
    assert a.confidence == 0.5  # one reader only: nothing agrees or disagrees
    assert cubicasa.answer(rel, "How many windows are there?").value == 0


def test_lone_box_counts_when_confident_enough(live):
    # the bedroom only the small model saw: 0.95 * 0.5 = 0.475 >= the bedrooms threshold (0.4) -> it counts
    rel = uploads.save(_png())
    a = cubicasa.answer(rel, "How many bedrooms are there?")
    assert a.value == 1 and len(a.evidence) == 1


def test_upload_then_ask_through_the_api(live):
    from stoa_qa.api import app

    c = TestClient(app)
    d = c.post("/api/upload", content=_png(), headers={"content-type": "image/png"}).json()
    assert d["id"].startswith("uploads/cubicasa5k/") and d["upload"] and d["source"] == "cubicasa5k"
    assert d["sheets"][0]["width"] == 400 and d["detector"]["available"] == ["m13-1280", "local-y11s-13c"]
    assert c.get(d["sheets"][0]["src"]).status_code == 200            # the sheet image itself is served
    assert c.get("/api/drawing", params={"id": d["id"]}).json()["id"] == d["id"]
    r = c.post("/api/ask", json={"id": d["id"], "question": "How many doors are there?"}).json()
    assert r["value"] == 3 and r["engine"]["kind"] == "detection" and r["engine"]["llm_tokens"] == 0
    assert r["overlay"]["space"] == "px" and len(r["overlay"]["items"]) == 3
    assert all(it["basis"] == "inferred" for it in r["overlay"]["items"])


def test_upload_rejects_non_images_and_never_serves_outside_the_uploads_folder(live):
    from stoa_qa.api import app

    c = TestClient(app)
    assert c.post("/api/upload", content=b"%PDF-1.4 nope", headers={"content-type": "image/png"}).status_code == 400
    assert c.post("/api/upload", content=b"").status_code == 400
    for bad in ("uploads/../../.env", "uploads/../uploads/x.png", "uploads/cubicasa5k/missing.png"):
        assert c.get("/api/file", params={"path": bad}).status_code == 404
    assert c.post("/api/ask", json={"id": "uploads/cubicasa5k/missing.png", "question": "How many doors?"}).status_code == 404


def test_upload_without_a_detector_is_refused_not_guessed(live, monkeypatch):
    from stoa_qa.api import app

    monkeypatch.setattr(cubicasa_live, "has_ultralytics", lambda: False)
    c = TestClient(app)
    d = c.post("/api/upload", content=_png(), headers={"content-type": "image/png"}).json()
    assert d["detector"]["available"] == []
    r = c.post("/api/ask", json={"id": d["id"], "question": "How many doors are there?"}).json()
    assert r["value"] is None  # STOA_NO_LLM: no Gemma fallback either, so no number


def test_a_sheet_the_detectors_cannot_read_is_declined_not_counted_as_zero(live, monkeypatch):
    # both models see one faint door on the whole sheet: a "0 windows" would be a blind guess, so the count step declines
    faint = {"m13-1280": [[DOOR, 0.2, 10, 10, 30, 20]], "local-y11s-13c": [[DOOR, 0.2, 10, 10, 30, 20]]}
    monkeypatch.setattr(cubicasa_live, "_load", lambda run: type("F", (), {
        "predict": lambda self, source, imgsz, conf, verbose: [_Result(faint[run])]})())
    rel = uploads.save(_png())
    a = cubicasa.answer(rel, "How many windows are there?")
    assert a.value is None and a.intent == "unreadable" and "cannot read this drawing" in a.method

    from stoa_qa import explain
    w = explain.explain(uploads.drawing(rel), "How many windows are there?", a, [], llm_ok=False)
    assert w["category"] == "unreadable" and w["alternatives"] == []


def test_tile_grid_covers_the_sheet_with_overlap():
    g = cubicasa_live.tile_grid(2382, 1684, 960)
    assert g[0][:2] == (0, 0) and max(x1 for *_, x1, _ in g) == 2382 and max(t[3] for t in g) == 1684
    assert all(t[2] - t[0] == 960 and t[3] - t[1] == 960 for t in g)
    assert cubicasa_live.tile_grid(500, 700, 960) == [(0, 0, 500, 700)]


def test_printed_room_labels_rescue_a_room_type_the_detector_missed(live, monkeypatch):
    # the mocked detectors see 3 doors and (fused) one weak bedroom; no kitchen. The plan prints two "Bedroom"s and a kitchen.
    from stoa_qa import room_labels

    lines = [{"box": [10, 10, 60, 20], "text": "Bedroom", "conf": 0.9}, {"box": [200, 10, 250, 20], "text": "Bedroom", "conf": 0.9},
             {"box": [100, 100, 150, 110], "text": "Kitchen", "conf": 0.9}, {"box": [100, 200, 160, 210], "text": "Living Room", "conf": 0.9}]
    monkeypatch.setattr(room_labels, "read", lambda image, key=None: lines)
    rel = uploads.save(_png())
    k = cubicasa.answer(rel, "How many kitchens are there?")
    # a blank test sheet has no walls to bound a room, so the label itself is the evidence
    assert k.value == 1 and "names 1 kitchen with printed labels" in k.method and k.evidence[0].coords == [[100, 100], [150, 110]]
    assert cubicasa.answer(rel, "How many bedrooms are there?").value == 1   # the detector found one: it is not overruled
    assert cubicasa.answer(rel, "How many saunas are there?").value == 0      # no sauna label: the detector's 0 stands


def test_room_label_vocabulary():
    from stoa_qa.room_labels import count, room_type

    assert [room_type(t) for t in ["Bedroom 2", "MH", "Makuuhuone", "Schlafzimmer", "WC/PH", "KPH", "Bathroom", "Keittiö",
                                   "Living Room", "Parveke"]] == \
        ["bedrooms", "bedrooms", "bedrooms", "bedrooms", "wc_rooms", "bathrooms", "bathrooms", "kitchens", None, "balconies"]
    assert count([{"text": "Bedroom"}], "bedrooms") is None  # one room name is not a labelled plan


def test_a_huge_sheet_is_stored_scaled_to_the_working_size(live):
    buf = io.BytesIO()
    Image.new("RGB", (10200, 6600), "white").save(buf, format="JPEG")
    rel = uploads.save(buf.getvalue())
    with Image.open(uploads.resolve(rel)) as im:
        assert im.size == (4000, 2588)


def test_uploads_use_the_two_style_detector_and_tile_only_when_it_finds_far_more(live, monkeypatch):
    # mixed-m13 installed: whole sheet sees 1 door, tiles see 12 -> tiles win; its own thresholds apply
    (live["weights"] / "mixed-m13.pt").write_bytes(b"x")
    cubicasa_live.live_config.cache_clear()
    monkeypatch.setattr(cubicasa_live, "predict_tiled", lambda model, image, size, tile, full=None:
                        [[DOOR, 0.9, 10 + 60 * i, 10, 40 + 60 * i, 40] for i in range(12)])
    monkeypatch.setattr(cubicasa_live, "_load", lambda run: type("M", (), {
        "predict": lambda self, source, imgsz, conf, verbose: [_Result([[DOOR, 0.9, 10, 10, 40, 40]])]})())
    big = io.BytesIO()
    Image.new("RGB", (2400, 1600), "white").save(big, format="PNG")
    rel = uploads.save(big.getvalue())
    out = cubicasa_live.boxes_live(uploads.resolve(rel))
    assert out["runs"] == ["mixed-m13"] and out["tiled"] and len(out["boxes"]) == 12
    a = cubicasa.answer(rel, "How many doors are there?")
    assert a.value == 12 and "in overlapping 960 px tiles" in a.method and "CubiCasa val + Swiss val" in a.method
    small = uploads.save(_png())  # 400 px: no tiling at all
    assert not cubicasa_live.boxes_live(uploads.resolve(small))["tiled"]


def test_a_colour_coded_sheet_counts_rooms_by_its_legend(live, monkeypatch):
    # a legend swatch + text, three rooms filled with that colour (one far away in a small inset), one other room
    import numpy as np
    from stoa_qa import sheet_ocr

    green = (200, 230, 200)  # RGB
    a = np.full((600, 1200, 3), 255, np.uint8)
    a[40:60, 900:960] = green                                   # legend swatch, left of its text
    for x0 in (100, 300, 500):
        a[200:330, x0:x0 + 150] = green                         # three dorm rooms of the plan
    a[200:330, 260:300] = (240, 200, 200)                       # a pink room between two of them
    a[540:560, 1100:1130] = green                               # a key-plan inset far away
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    rel = uploads.save(buf.getvalue())
    monkeypatch.setattr(sheet_ocr, "read", lambda image, key=None: [{"box": [970, 38, 1080, 62], "text": "DORMITORY", "conf": 0.9}])
    ans = cubicasa.answer(rel, "How many bedrooms are there?")
    assert ans.value == 3 and "'DORMITORY'" in ans.method and len(ans.evidence) == 3
    assert all(e.kind == "polygon" for e in ans.evidence)
