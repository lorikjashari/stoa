"""cubicasa5k raster plans: answer from per-plan predictions of several sources (the hybrid).

Which source answers which question is decided on CubiCasa's val split by stoa_qa.cubicasa_hybrid and
stored in stoa_qa/cubicasa_picks.json. Evidence:
  yolo     one box per counted object (pixel coords on the plan image)
  network  room / icon polygons from the CubiCasa5K network
  gemma / gemini  the model's count only (no geometry): marked as such, lower confidence
Confidence comes from agreement between independent sources.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

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
READER = {"network": "the CubiCasa network", "gemma": "Gemma 4", "gemini": "Gemini 3.5 Flash", "yolo": "the detector"}
YOLO_CLASSES = ["door", "window", "toilet", "sink", "shower", "bathtub"]
YOLO_Q = {"doors": ["door"], "windows": ["window"], "toilets": ["toilet"], "sinks": ["sink"], "baths": ["shower", "bathtub"]}
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


def predictions(stem: str) -> dict[str, dict]:
    """All available source predictions for one plan: {source: {"counts": {...}, ...}}."""
    out = {}
    net = _json(REPO / "third_party/cubicasa_preds" / f"{stem}.json")
    if net and "counts" in net:
        out["network"] = net
    for src, d in (("gemma", "google__gemma-4-31B-it"), ("gemini", "gemini-3.5-flash")):
        r = _json(REPO / "eval_out/gemini" / d / f"{stem}.json")
        if r and "counts" in r:
            out[src] = r
    cfg = picks()
    if cfg.get("yolo_run"):
        boxes = _yolo_boxes(cfg["yolo_run"]).get(stem)
        if boxes is not None:
            thr = cfg["yolo_thresholds"]
            counts = {q: sum(1 for b in boxes if YOLO_CLASSES[b[0]] in cls and b[1] >= thr.get(q, 0.5))
                      for q, cls in YOLO_Q.items()}
            out["yolo"] = {"counts": counts, "boxes": boxes}
    return out


def _evidence(src: str, key: str, pred: dict, cfg: dict) -> list[Evidence]:
    if src == "yolo":
        thr = cfg["yolo_thresholds"].get(key, 0.5)
        sel = [b for b in pred["boxes"] if YOLO_CLASSES[b[0]] in YOLO_Q[key] and b[1] >= thr]
        return [Evidence("box", f"yolo-{i}", f"{YOLO_CLASSES[b[0]]} {i + 1}", b[1], [[b[2], b[3]], [b[4], b[5]]])
                for i, b in enumerate(sel)]
    if src == "network":
        if key in NET_ROOM:
            rooms = [r for r in pred.get("rooms", []) if r["class"] == NET_ROOM[key]]
            return [Evidence("polygon", f"net-room-{i}", f"{LABELS[key]} {i + 1}", None, r["poly"]) for i, r in enumerate(rooms)]
        if key in NET_ICON:
            icons = [r for r in pred.get("icons", []) if r["class"] in NET_ICON[key]]
            return [Evidence("polygon", f"net-icon-{i}", f"{LABELS[key]} {i + 1}", None, r["poly"]) for i, r in enumerate(icons)]
        return []
    return [Evidence("model", src, f"{src} counted {pred['counts'].get(key)} (no geometry)")]


def answer(path: str, question: str) -> Answer:
    stem = Path(path).stem
    key = question_key(question)
    if key is None:
        return Answer(None, None, "unknown", 0.0, "cubicasa: question not understood", source_file=path)
    preds = predictions(stem)
    cfg = picks()
    order = [cfg["picks"].get(key)] + ["yolo", "gemini", "gemma", "network"]
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
    ev = _evidence(src, key, preds[src], cfg)
    if not any(e.coords for e in ev):
        for helper in ("yolo", "network"):
            if helper in preds and helper != src:
                shapes = [e for e in _evidence(helper, key, preds[helper], cfg) if e.coords]
                if shapes:
                    ev = shapes
                    method += f" The marks on the sheet are {READER[helper]}'s {len(shapes)} detections."
                    break
    return Answer(value, "count", key, conf, method, ev, path)
