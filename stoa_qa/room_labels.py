"""Room-type counts read from the room names printed on a raster plan (OCR, no LLM).

The plan detectors learned room types (bedroom, kitchen, bathroom, WC, sauna, balcony) from CubiCasa's Finnish plans,
where e.g. a bedroom is a space marked "MH". On a plan drawn elsewhere that prints "Bedroom" they often see nothing,
yet the printed name is the plan's own statement of what the room is, and it can be checked on the sheet. So for an
uploaded plan that labels its rooms, the room-type count is the number of printed labels of that type, each label's
box is the evidence, and the detector's count is shown beside it as a cross-check.

The text comes from stoa_qa.sheet_ocr (RapidOCR, cached per image, shared with the document clerk). Matching is plain keyword patterns in EN/FI/DE/FR/IT/SV/NL/ES; a "WC/PH" line counts as
a WC room only, as in CubiCasa's own rule (stoa_qa.cubicasa_rules.room_of).
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MIN_LABELLED = 2  # printed room names a plan needs before "no label of this type" is read as "none"

ROOM_TYPES = {
    "wc_rooms": r"^w\.? ?c\b|\bwc\b|\btoilet\b(?! bowl)|\bpowder room|\bvessa|\bwc-tila",
    "bedrooms": r"\bbed ?rooms?\b|\bmaster bed|\bbdrm\b|^br ?\d?$|^\d? ?mh ?\d?$|\bmh\d|makuuhuone|schlafzimmer|"
                r"\bchambre\b|camera da letto|\bsovrum|slaapkamer|dormitorio|\bhabitacion|sypialnia",
    "kitchens": r"\bkitchen|\bkitchenette|keittio|^kk$|^kt$|^k$|\bkuche|\bkueche|\bcuisine|\bcucina|^kok$|\bkeuken|\bcocina",
    "bathrooms": r"\bbath ?rooms?\b|^bath$|\bwash ?room|^kh\b|^kph\b|^ph\b|\bpesuh|^sh\b|kylpyhuone|pesuhuone|^bad\b|"
                 r"\bsalle de bains?\b|\bbagno\b|\bbadrum|\bbadkamer|\bbano\b",
    "saunas": r"\bsauna|\bbastu",
    "balconies": r"\bbalcon|\bbalkon|\bparveke|^parv\b|\bloggia",
}
# other room names: a plan printing these labels its rooms, so a missing bedroom label means no bedroom
OTHER_ROOMS = (r"\bliving|\blounge|\bdining|\bhall\b|\bhallway|\bfoyer|\bentry|\bporch|\blaundry|\butility|\bcloset|"
               r"\bstudy\b|\boffice\b|\bfamily room|\bgarage|\bstorage|\bpantry|\bwardrobe|\bwalk-?in|"
               r"^oh$|olohuone|^et$|eteinen|^vh$|^var$|^ruok|^kod$|^tk$|^khh$|^at$|^as$|"
               r"\bwohnen|\bwohnzimmer|\bkorridor|\bflur\b|\bsejour|\bcouloir|\bsoggiorno|\bcorridoio")

def norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9.\-/ ]", " ", t).split())


def room_type(text: str) -> str | None:
    """The room type a printed label names, or None. WC lines are WC rooms only (CubiCasa's rule)."""
    t = norm(text)
    return next((k for k, pat in ROOM_TYPES.items() if re.search(pat, t)), None)


def is_room_name(text: str) -> bool:
    return room_type(text) is not None or re.search(OTHER_ROOMS, norm(text)) is not None


def read(image: str | Path, key: str | None = None) -> list[dict] | None:
    """Printed text lines of the sheet (stoa_qa.sheet_ocr: one pass for a small plan, tiles for a large sheet)."""
    from . import sheet_ocr

    return sheet_ocr.read(image, key)


def count(lines: list[dict], key: str) -> dict | None:
    """{'labels': [lines naming a room of this type], 'room_names': n} or None when the plan prints no room names
    (then the labels say nothing, and the detector's count stands)."""
    names = [ln for ln in lines if is_room_name(ln["text"])]
    if len(names) < MIN_LABELLED:
        return None
    return {"labels": [ln for ln in names if room_type(ln["text"]) == key], "room_names": len(names)}


def rooms_of(image, labels: list[dict]) -> list[dict]:
    """The walled room each printed label sits in: [{'labels': [...], 'poly': [[x, y], ...] | None, 'bbox'}].

    Walls are the thick dark strokes (thin furniture and text lines are opened away), door and window gaps are bridged
    along the wall axes, and a label's room is the free space around it. Labels in one room give one room. A label whose
    space leaks to the sheet edge or covers over a third of the sheet (an open plan, a porch) keeps its label box."""
    import cv2
    import numpy as np

    img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [{"labels": [ln], "poly": None, "bbox": ln["box"]} for ln in labels]
    h, w = img.shape
    dark = (img < 110).astype(np.uint8)
    dt = cv2.distanceTransform(dark, cv2.DIST_L2, 3)
    thick = int(round(2 * np.percentile(dt[dt > 0], 97))) if (dt > 0).any() else 0
    if thick < 4:  # no thick walls to bound rooms (thin-line drawing): labels only
        return [{"labels": [ln], "poly": None, "bbox": ln["box"]} for ln in labels]
    walls = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((max(3, thick // 2),) * 2, np.uint8))
    g = thick * 6
    closed = walls.copy()
    for kern in (np.ones((1, g), np.uint8), np.ones((g, 1), np.uint8)):
        closed |= cv2.morphologyEx(walls, cv2.MORPH_CLOSE, kern)
    n, lab, st, _ = cv2.connectedComponentsWithStats((closed == 0).astype(np.uint8), 4)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    rooms: dict[int, dict] = {}
    out = []
    for ln in labels:
        x0, y0, x1, y1 = (int(v) for v in ln["box"])
        win = lab[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4]
        ids = win[win > 0]
        i = int(np.bincount(ids).argmax()) if ids.size else 0
        if i and i not in edge and st[i, 4] < h * w / 3:
            if i not in rooms:
                cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c = max(cs, key=cv2.contourArea)
                poly = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True).reshape(-1, 2).tolist()
                x, y, bw, bh = (int(v) for v in st[i, :4])
                rooms[i] = {"labels": [], "poly": poly, "bbox": [x, y, x + bw, y + bh]}
                out.append(rooms[i])
            rooms[i]["labels"].append(ln)
        else:
            out.append({"labels": [ln], "poly": None, "bbox": ln["box"]})
    return out
