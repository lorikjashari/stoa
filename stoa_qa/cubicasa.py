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
    ("toilets", r"toilet bowl|wc seat|wc-istuin|\btoilets?\b(?! ?rooms?)"),  # plain "toilets" too; "toilet rooms" = wc_rooms
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

        # uploads: the detector trained on two drawing styles when installed (cubicasa_live_picks.json), with its own
        # thresholds; else the dataset's fused pair with the dataset's thresholds
        live = cubicasa_live.boxes_live(image) or cubicasa_live.boxes(cfg["yolo_run"], image)
        if live is not None:
            lcfg = live.get("cfg") or cfg
            boxes = live["boxes"]
            info = {k: live.get(k) for k in ("runs", "fused", "complete", "hash", "tiled", "selected_on")}
            from . import plan3d

            if plan3d.available():  # doors and windows must sit in a wall: on a new drawing, others are misread furniture
                info["openings"] = plan3d.openings_of(boxes, lcfg)
                off = plan3d.off_wall(image, boxes, lcfg)
                if off:  # scored 0, not removed: a box's index stays its 3D id
                    boxes = [[b[0], 0.0, *b[2:]] if k in off else b for k, b in enumerate(boxes)]
                    info["off_wall"] = len(off)
            counts = {q: len(_yolo_selected(boxes, q, lcfg)) for q in YOLO_Q}
            out["yolo"] = {"counts": counts, "boxes": boxes, "cfg": lcfg, "live": info}
    return out


def _yolo_selected(boxes: list, key: str, cfg: dict, pre: str = "yolo") -> list:
    """The detector boxes counted for one question, with the threshold and overlap cut chosen on val."""
    return rules.select_boxes(boxes, key, (cfg.get(f"{pre}_thresholds") or {}).get(key, 0.5),
                              (cfg.get(f"{pre}_iou") or {}).get(key, 1.0))


def _evidence(src: str, key: str, pred: dict, cfg: dict) -> list[Evidence]:
    if src in DETECTORS:
        sel = _yolo_selected(pred["boxes"], key, cfg, DETECTORS[src])
        pos = {id(b): k for k, b in enumerate(pred["boxes"])}  # a box's index in the plan's boxes is its 3D id (stoa_qa.plan3d)
        return [Evidence("box", f"{src}-{i}", f"{YOLO_CLASSES[b[0]]} {i + 1}", b[1], [[b[2], b[3]], [b[4], b[5]]],
                         viewer_id=f"{src}:{pos[id(b)]}") for i, b in enumerate(sel)]
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
    note = (f"Detected on this image just now by YOLO11 ({runs}, {how}), with the thresholds chosen on "
            f"{live.get('selected_on') or 'CubiCasa' + chr(39) + 's val split'}")
    if live.get("tiled"):
        note += ", in overlapping 960 px tiles because its symbols are small for the whole sheet"
    if not live["complete"]:
        note += f" for {cfg.get('yolo_run')}; only part of that run is installed here, so the count is less certain"
    note += "."
    if live.get("off_wall"):
        n = live["off_wall"]
        note += f" {n} door/window detection{'s' if n != 1 else ''} not set in any wall {'were' if n != 1 else 'was'} left out (furniture misread)."
    return note


OBJECT_QUESTIONS = ("doors", "windows", "toilets", "sinks", "baths")
MIN_READ = 3  # confident objects a detector must find on a whole plan before its counts (zeros included) are trusted


def readable(live_boxes: list, cfg: dict) -> int:
    """Confident doors, windows and fixtures the detector found on the whole sheet, with the val-tuned thresholds.

    A real floor plan has several. Almost none means the drawing is outside what the detector learned (another drawing
    style, or symbols far smaller than in its training plans), and then a count of 0 is a blind guess, not a finding."""
    return sum(len(_yolo_selected(live_boxes, q, cfg)) for q in OBJECT_QUESTIONS)


def answer(path: str, question: str) -> Answer:
    stem = Path(path).stem
    key = question_key(question)
    if key is None:
        return Answer(None, None, "unknown", 0.0, "cubicasa: question not understood", source_file=path)
    if path.startswith("uploads/") and key in LABELLED_ROOMS:
        by_legend = _legend_answer(path, key)  # a colour-coded sheet: its legend says which drawn rooms are which
        if by_legend is not None:
            return by_legend
    preds = predictions_for(path)
    cfg = picks()
    live = (preds.get("yolo") or {}).get("live")
    ycfg = (preds.get("yolo") or {}).get("cfg") or cfg  # the thresholds of whichever detector produced the boxes
    if live and (n := readable(preds["yolo"]["boxes"], ycfg)) < MIN_READ:
        return Answer(None, "count", "unreadable", 0.0,
                      f"The plan detectors cannot read this drawing: they found {n} confident door, window or fixture "
                      f"symbol{'s' if n != 1 else ''} on the whole sheet. They learned from residential plans; this style or "
                      f"symbol size is outside that, so a count would be a guess.",
                      source_file=path)
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
    if live and src == "yolo":
        method = f"{method.rstrip('.')}. {_live_note(live, cfg)}"
    ev = _evidence(src, key, preds[src], ycfg if src == "yolo" else cfg)
    if not any(e.coords for e in ev):
        for helper in ("tiles", "yolo", "network"):
            if helper in preds and helper != src:
                shapes = [e for e in _evidence(helper, key, preds[helper], ycfg if helper == "yolo" else cfg) if e.coords]
                if shapes:
                    ev = shapes
                    method += f" The marks on the sheet are {READER[helper]}'s {len(shapes)} detections."
                    break
    if live and value == 0 and key in LABELLED_ROOMS:
        rescued = _label_rescue(path, key, live)
        if rescued is not None:
            return rescued
    return Answer(value, "count", key, conf, method, ev, path)


LABELLED_ROOMS = ("bedrooms", "kitchens", "bathrooms", "wc_rooms", "saunas", "balconies")


def _legend_answer(path: str, key: str) -> Answer | None:
    """Rooms of this type counted by their legend colour (stoa_qa.legend_rooms); None when the sheet has no colour legend
    entry for the type (or no OCR), so the detector and the printed labels take over."""
    from . import cubicasa_live, legend_rooms, sheet_ocr

    image = image_of(path)
    if image is None:
        return None
    try:
        lines = sheet_ocr.read(image, cubicasa_live.image_hash(image))
        r = legend_rooms.count(image, lines, key) if lines else None
    except Exception:
        return None
    if r is None:
        return None
    rooms, n = r["rooms"], len(r["rooms"])
    per = {}
    for rm in rooms:
        per[rm["legend"]] = per.get(rm["legend"], 0) + 1
    parts = " + ".join(f"{c} '{lg}'" for lg, c in per.items()) or "none"
    legends = ", ".join(f"'{e['text']}'" for e in r["legends"])
    method = (f"Counted {n} {NOUNS[key][n != 1]} drawn on the plan: the rooms filled with the legend colour of {legends} "
              f"({parts}). Each counted room is outlined on the sheet; insets such as a key plan are left out.")
    ev = [Evidence("polygon", f"legend-room-{i}", f"{rm['legend']} {i + 1}", None, rm["poly"]) for i, rm in enumerate(rooms)]
    return Answer(n, "count", key, 0.8, method, ev, path)


def _label_rescue(path: str, key: str, live: dict) -> Answer | None:
    """The detector found no room of this type, but the plan prints labels naming such rooms: count the labels.

    Only as a rescue: on CubiCasa val (_analysis/room_labels_eval.py) printed labels alone are far worse than the detector
    (OCR misreads short Finnish codes such as MH), while "labels only when the detector saw none" is never worse on any
    room type (bedrooms 88.5 = 88.5%, balconies 93.0 -> 94.5%). It catches plans drawn unlike CubiCasa's that print
    "Bedroom"."""
    from . import room_labels

    image = image_of(path)
    lines = room_labels.read(image, key=live.get("hash")) if image is not None else None
    c = room_labels.count(lines, key) if lines else None
    if not c or not c["labels"]:
        return None
    rooms = room_labels.rooms_of(image, c["labels"])  # the walled room each label names; two labels in one room = one
    n = len(rooms)
    names = ", ".join(f"'{ln['text']}'" for ln in c["labels"][:6])
    walled = sum(r["poly"] is not None for r in rooms)
    method = (f"The detector found no {NOUNS[key][1]}, but the plan names {n} {NOUNS[key][n != 1]} with printed labels "
              f"({names})" + (f"; {walled} of them {'is' if walled == 1 else 'are'} outlined as the walled room the label "
                              f"sits in" if walled else "; each label is marked") + ".")
    ev = [Evidence("polygon", f"label-room-{i}", f"{LABELS[key]} {i + 1} ('{r['labels'][0]['text']}')", None, r["poly"])
          if r["poly"] is not None else
          Evidence("box", f"label-{i}", f"printed '{r['labels'][0]['text']}'", r["labels"][0].get("conf"),
                   [[r["bbox"][0], r["bbox"][1]], [r["bbox"][2], r["bbox"][3]]])
          for i, r in enumerate(rooms)]
    return Answer(n, "count", key, 0.7, method, ev, path)
