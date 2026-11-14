"""Segmentation connected components vs polygon counts, on a val sample.

The published polygons are built from junction points, which drop most bathtubs and merge openings.
The class maps themselves may count more faithfully.
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "blueprint3d"))
os.environ.setdefault("STOA_DATASET", str(REPO / "junction-dataset" / "junction-dataset"))

from cubicasa_model import ICON_CLASSES, ROOM_CLASSES, predict  # noqa: E402
from stoa_qa.cubicasa_gemini import load_truth, split_members  # noqa: E402
from stoa_qa.core import DATASET  # noqa: E402

KEYS_ICON = {
    "doors": "Door",
    "windows": "Window",
    "toilets": "Toilet",
    "sinks": "Sink",
    "baths": "Bathtub",
}
KEYS_ROOM = {
    "kitchens": "Kitchen",
    "bedrooms": "Bed Room",
    "bathrooms": "Bath",
    "balconies": "Outdoor",
    "saunas": "Sauna Bench",  # icon, handled below
}


def components(mask: np.ndarray, min_area: int) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return sum(int(stats[i, cv2.CC_STAT_AREA]) >= min_area for i in range(1, n))


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    truth = load_truth()
    preds = {}
    for f in (REPO / "third_party/cubicasa_preds").glob("*.json"):
        r = json.loads(f.read_text())
        if "counts" in r:
            preds[r["path"]] = r["counts"]
    members = sorted(p for p in split_members("val") if p in truth and p in preds)[:limit]
    print("sample", len(members), flush=True)

    rows = []
    t0 = time.time()
    for i, rel in enumerate(members):
        t = time.time()
        hm, rooms, icons, scale = predict(str(DATASET / rel))
        icon_map = icons.argmax(0)
        room_map = rooms.argmax(0)
        icon_p = icons.max(0)
        rows.append((rel, rooms, icons, icon_map, room_map, icon_p, scale))
        print(f"  {i + 1}/{len(members)} {Path(rel).name} {time.time() - t:.1f}s", flush=True)
    print(f"inference {time.time() - t0:.0f}s", flush=True)

    def score(name, getter):
        ok = n = 0
        per = {k: [0, 0] for k in list(KEYS_ICON) + ["kitchens", "bedrooms", "bathrooms", "balconies", "saunas"]}
        for rel, rooms, icons, icon_map, room_map, icon_p, scale in rows:
            counts = getter(rel, rooms, icons, icon_map, room_map, icon_p, scale)
            for k, pred in counts.items():
                if k not in truth[rel]:
                    continue
                per[k][1] += 1
                hit = int(pred) == truth[rel][k]
                per[k][0] += hit
                ok += hit
                n += hit or True
        print(f"\n{name}")
        for k, (a, b) in per.items():
            if b:
                print(f"  {k:12} {a / b:6.1%}  n={b}")
        print(f"  {'ALL':12} {ok / max(n, 1):6.1%}  n={n}")

    def polygon_counts(rel, *rest):
        return {k: preds[rel][k] for k in preds[rel] if k in (
            "doors", "windows", "toilets", "sinks", "baths", "saunas", "kitchens", "bedrooms", "bathrooms", "balconies")}

    def cc_counts(min_icon, min_room, conf):
        def getter(rel, rooms, icons, icon_map, room_map, icon_p, scale):
            out = {}
            for key, cls in KEYS_ICON.items():
                idx = ICON_CLASSES.index(cls)
                mask = (icon_map == idx) & (icon_p >= conf)
                out[key] = components(mask, min_icon)
            for key, cls in (("kitchens", "Kitchen"), ("bedrooms", "Bed Room"), ("bathrooms", "Bath"), ("balconies", "Outdoor")):
                idx = ROOM_CLASSES.index(cls)
                out[key] = components(room_map == idx, min_room)
            idx = ICON_CLASSES.index("Sauna Bench")
            out["saunas"] = components((icon_map == idx) & (icon_p >= conf), min_icon)
            return out
        return getter

    score("polygons (cached)", polygon_counts)
    for min_icon, min_room, conf in ((15, 80, 0.3), (30, 150, 0.4), (50, 250, 0.5), (80, 400, 0.5)):
        score(f"cc icon>={min_icon} room>={min_room} p>={conf}", cc_counts(min_icon, min_room, conf))


if __name__ == "__main__":
    main()
