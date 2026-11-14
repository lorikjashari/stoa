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
