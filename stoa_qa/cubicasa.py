"""cubicasa5k raster plans: answer from per-plan predictions of several sources (the hybrid).

Which source answers which question is decided on CubiCasa's val split by stoa_qa.cubicasa_hybrid and
stored in stoa_qa/cubicasa_picks.json. Evidence:
  yolo     one box per counted object (pixel coords on the plan image)
  tiles    the same, from the high-resolution door/window detector (modal_jobs/cubicasa_tiles.py)
  network  room / icon polygons from the CubiCasa5K network
  gemma / gemini  the model's count only (no geometry): marked as such, lower confidence
Confidence comes from agreement between independent sources.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from . import cubicasa_refine as refine
from . import cubicasa_rules as rules
from .core import Answer, Evidence

REPO = Path(__file__).resolve().parent.parent
PICKS = Path(__file__).with_name("cubicasa_picks.json")

# order matters: e.g. the sinks question mentions kitchens and bathrooms, the bathrooms question mentions WC rooms
KEY_PATTERNS = [
    ("sinks", r"\bsinks?\b|wash ?basin|allas"),
    ("baths", r"shower|bathtub|suihku|kylpyamme"),
    ("toilets", r"toilet bowl|wc seat|wc-istuin"),
    ("bathrooms", r"bathroom|washroom|pesuhuone|kylpyhuone"),
    ("wc_rooms", r"wc rooms?|toilet rooms?|wc-tila"),
    ("kitchens", r"kitchen|keittiö"),
    ("bedrooms", r"bedroom|makuuhuone"),
    ("saunas", r"sauna"),
    ("balconies", r"balcon|parvek"),
    ("doors", r"\bdoors?\b|\bovea\b|\bovi"),
    ("windows", r"window|ikkun"),
    ("floors", r"\bfloors\b|storeys?|kerrosta|kerroksia"),
]
LABELS = {"doors": "door", "windows": "window", "floors": "floor", "toilets": "toilet", "sinks": "sink",
          "baths": "shower/bathtub", "saunas": "sauna", "kitchens": "kitchen", "bedrooms": "bedroom",
          "wc_rooms": "WC room", "bathrooms": "bathroom", "balconies": "balcony"}
NOUNS = {"doors": ("door", "doors"), "windows": ("window", "windows"), "floors": ("floor", "floors"), "toilets": ("toilet bowl", "toilet bowls"),
         "sinks": ("sink", "sinks"), "baths": ("shower or bathtub", "showers and bathtubs"), "saunas": ("sauna", "saunas"),
         "kitchens": ("kitchen", "kitchens"), "bedrooms": ("bedroom", "bedrooms"), "wc_rooms": ("separate WC room", "separate WC rooms"),
         "bathrooms": ("bathroom", "bathrooms"), "balconies": ("balcony", "balconies")}
READER = {"network": "the CubiCasa network", "gemma": "Gemma 4", "gemini": "Gemini 3.5 Flash", "yolo": "the detector",
          "tiles": "the high-resolution door/window detector"}
DETECTORS = {"yolo": "yolo", "tiles": "tile"}  # source -> key prefix in cubicasa_picks.json (<prefix>_run, _thresholds, _iou)
TILE_QUESTIONS = ("doors", "windows")
YOLO_CLASSES = rules.CLASSES      # the detector is trained on these classes (stoa_qa.cubicasa_rules)
YOLO_Q = rules.QUESTION_CLASSES   # question -> detector classes counted for it
NET_ROOM = {"kitchens": "Kitchen", "bedrooms": "Bed Room", "bathrooms": "Bath", "balconies": "Outdoor"}
NET_ICON = {"doors": ["Door"], "windows": ["Window"], "toilets": ["Toilet"], "sinks": ["Sink"], "baths": ["Bathtub"],
            "saunas": ["Sauna Bench"]}


def question_key(q: str) -> str | None:
    ql = q.lower()
    return next((k for k, pat in KEY_PATTERNS if re.search(pat, ql)), None)


@lru_cache(maxsize=1)
def picks() -> dict:
    return json.loads(PICKS.read_text()) if PICKS.exists() else {"picks": {}, "yolo_run": None, "yolo_thresholds": {}}


@lru_cache(maxsize=4)
def _yolo_boxes(run: str) -> dict:
    out = {}
    for split in ("val", "test", "all"):
        f = REPO / "eval_out/yolo" / run / f"{split}.json"
        if f.exists():
            out.update(json.loads(f.read_text()))
    return out


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


@lru_cache(maxsize=1)
def _seg_index() -> dict[str, dict]:
    """stem -> segmentation components, written by _analysis/raster_cache_seg.py."""
    f = REPO / "eval_out" / "seg_components.jsonl"
    out = {}
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[Path(rec["path"]).stem] = rec
    return out


def image_of(path: str) -> Path | None:
    """The plan image file behind a dataset- or upload-relative path (None when it is not on disk)."""
    from . import sheets

    try:
        return sheets.safe_path(path)
    except (FileNotFoundError, OSError, ValueError):
        return None


def predictions_for(path: str) -> dict[str, dict]:
    """predictions() for a plan given by path, so a plan without precomputed boxes can be detected live."""
    return predictions(Path(path).stem, image_of(path))


def predictions(stem: str, image: Path | None = None) -> dict[str, dict]:
    """All available source predictions for one plan: {source: {"counts": {...}, ...}}.

    Without precomputed detector boxes for `stem`, the picked YOLO run is executed on `image` when its weights and
    ultralytics are installed (stoa_qa.cubicasa_live); those boxes carry a "live" entry saying which models ran."""
    out = {}
    net = _json(REPO / "third_party/cubicasa_preds" / f"{stem}.json")
    if net and "counts" in net:
        seg = _seg_index().get(stem)
        net = {**net, "counts": refine.apply(net, seg)}
        if seg and seg.get("boxes"):
            net["boxes"] = seg["boxes"]
        out["network"] = net
    for src, d in (("gemma", "google__gemma-4-31B-it"), ("gemini", "gemini-3.5-flash")):
        r = _json(REPO / "eval_out/gemini" / d / f"{stem}.json")
        if r and "counts" in r:
            out[src] = r
    cfg = picks()
    for src, pre in DETECTORS.items():
        if cfg.get(f"{pre}_run"):
            boxes = _yolo_boxes(cfg[f"{pre}_run"]).get(stem)
            if boxes is not None:
                qs = TILE_QUESTIONS if src == "tiles" else YOLO_Q
                counts = {q: len(_yolo_selected(boxes, q, cfg, pre)) for q in qs}
                out[src] = {"counts": counts, "boxes": boxes}
    if "yolo" not in out and image is not None and cfg.get("yolo_run"):
        from . import cubicasa_live

        live = cubicasa_live.boxes(cfg["yolo_run"], image)
        if live is not None:  # same format and thresholds as the precomputed boxes of the picked run
            counts = {q: len(_yolo_selected(live["boxes"], q, cfg)) for q in YOLO_Q}
            out["yolo"] = {"counts": counts, "boxes": live["boxes"],
                           "live": {k: live[k] for k in ("runs", "fused", "complete", "hash")}}
    return out


def _yolo_selected(boxes: list, key: str, cfg: dict, pre: str = "yolo") -> list:
    """The detector boxes counted for one question, with the threshold and overlap cut chosen on val."""
    return rules.select_boxes(boxes, key, (cfg.get(f"{pre}_thresholds") or {}).get(key, 0.5),
                              (cfg.get(f"{pre}_iou") or {}).get(key, 1.0))


def _evidence(src: str, key: str, pred: dict, cfg: dict) -> list[Evidence]:
    if src in DETECTORS:
        sel = _yolo_selected(pred["boxes"], key, cfg, DETECTORS[src])
        return [Evidence("box", f"{src}-{i}", f"{YOLO_CLASSES[b[0]]} {i + 1}", b[1], [[b[2], b[3]], [b[4], b[5]]])
                for i, b in enumerate(sel)]
    if src == "network":
        boxes = (pred.get("boxes") or {}).get(key)
        if boxes:
            return [Evidence("box", f"seg-{key}-{i}", f"{LABELS[key]} {i + 1}", None, b) for i, b in enumerate(boxes)]
        rooms = pred.get("rooms") or []
        scale = float(pred.get("scale") or 1.0)
        if key == "floors":
            return [Evidence("box", f"floor-{i}", f"floor {i + 1}", None, b) for i, b in enumerate(refine.floor_plates(rooms))]
        if key in ("baths", "wc_rooms", "bathrooms"):
            want = {"baths": None, "wc_rooms": "wc", "bathrooms": "bathroom"}[key]
            picked = [r for kind, r in refine.wet_kinds(rooms, scale) if want is None or kind == want]
            return [Evidence("polygon", f"wet-{i}", f"{LABELS[key]} {i + 1}", None, r["poly"]) for i, r in enumerate(picked)]
        # doors, windows, sinks and bedrooms are counted from the class map; only its boxes are the evidence
        if key in ("doors", "windows", "sinks", "bedrooms"):
            return []
        if key in NET_ROOM:
            found = [r for r in rooms if r["class"] == NET_ROOM[key]]
            return [Evidence("polygon", f"net-room-{i}", f"{LABELS[key]} {i + 1}", None, r["poly"]) for i, r in enumerate(found)]
        if key in NET_ICON:
            icons = [r for r in pred.get("icons", []) if r["class"] in NET_ICON[key]]
            return [Evidence("polygon", f"net-icon-{i}", f"{LABELS[key]} {i + 1}", None, r["poly"]) for i, r in enumerate(icons)]
        return []
    return [Evidence("model", src, f"{src} counted {pred['counts'].get(key)} (no geometry)")]


def _live_note(live: dict, cfg: dict) -> str:
    runs = " + ".join(live["runs"])
    how = "fused" if live["fused"] else "single model"
    note = f"Detected on this image just now by YOLO11 ({runs}, {how}), with the thresholds chosen on CubiCasa's val split"
    if not live["complete"]:
        note += f" for {cfg.get('yolo_run')}; only part of that run is installed here, so the count is less certain"
    return note + "."


def answer(path: str, question: str) -> Answer:
    stem = Path(path).stem
    key = question_key(question)
    if key is None:
        return Answer(None, None, "unknown", 0.0, "cubicasa: question not understood", source_file=path)
    preds = predictions_for(path)
    cfg = picks()
    order = [cfg["picks"].get(key)] + ["tiles", "yolo", "gemini", "gemma", "network"]
    src = next((s for s in order if s and s in preds and key in preds[s]["counts"]), None)
    if src is None:
        return Answer(None, "count", key, 0.0, f"no prediction available for {stem} (run the sources first)", source_file=path)
    value = int(preds[src]["counts"][key])
    votes = [int(p["counts"][key]) for s, p in preds.items() if key in p["counts"]]
    agree = votes.count(value) - 1
    conf = 0.9 if agree >= 1 else 0.5
    others = "; ".join(f"{READER[s]}: {int(p['counts'][key])}" for s, p in preds.items() if s != src and key in p["counts"])
    reader = READER[src][0].upper() + READER[src][1:]
    method = f"{reader} counted {value} {NOUNS[key][value != 1]}"
    if others:
        method += f". Other readers: {others}. " + ("All agree." if agree == len(votes) - 1 else f"{agree} of {len(votes) - 1} agree{'s' if agree == 1 else ''}." if agree else "None agrees, so check the marks.")
    live = (preds.get("yolo") or {}).get("live")
    if live and src == "yolo":
        method = f"{method.rstrip('.')}. {_live_note(live, cfg)}"
    ev = _evidence(src, key, preds[src], cfg)
    if not any(e.coords for e in ev):
        for helper in ("tiles", "yolo", "network"):
            if helper in preds and helper != src:
                shapes = [e for e in _evidence(helper, key, preds[helper], cfg) if e.coords]
                if shapes:
                    ev = shapes
                    method += f" The marks on the sheet are {READER[helper]}'s {len(shapes)} detections."
                    break
    return Answer(value, "count", key, conf, method, ev, path)
