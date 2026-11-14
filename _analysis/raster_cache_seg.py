"""Cache per-component areas from the CubiCasa class maps so thresholds can be chosen without re-running the network.

Writes eval_out/seg_components.jsonl (one plan per line). Safe to resume.
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "blueprint3d"))
os.environ.setdefault("STOA_DATASET", str(REPO / "junction-dataset" / "junction-dataset"))

from cubicasa_model import ICON_CLASSES, ROOM_CLASSES, predict  # noqa: E402
from stoa_qa.core import DATASET  # noqa: E402
from stoa_qa.cubicasa_gemini import split_members  # noqa: E402
from stoa_qa.cubicasa_refine import BED_MIN_AREA, ICON_MIN_AREA, SEG_QUESTIONS, boxes_from_mask  # noqa: E402

OUT = REPO / "eval_out" / "seg_components.jsonl"


def areas(mask: np.ndarray) -> list[int]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return sorted(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n))


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    done = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["path"])
    members = sorted(split_members(split))
    todo = [p for p in members if p not in done and (DATASET / p).is_file()]
    print(f"{split}: {len(members)} plans, {len(done)} cached, {len(todo)} to run", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with OUT.open("a", encoding="utf-8") as fp:
        for i, rel in enumerate(todo):
            t = time.time()
            _, rooms, icons, scale = predict(str(DATASET / rel))
            icon_map = icons.argmax(0)
            room_map = rooms.argmax(0)
            icon_p = icons.max(0)
            rec = {"path": rel, "scale": scale, "icons": {}, "rooms": {}}
            for name_i, name in enumerate(ICON_CLASSES):
                if name == "No Icon":
                    continue
                rec["icons"][name] = areas((icon_map == name_i) & (icon_p >= 0.35))
            for name_i, name in enumerate(ROOM_CLASSES):
                if name in ("Background", "Wall", "Railing"):
                    continue
                rec["rooms"][name] = areas(room_map == name_i)
            boxes = {}
            for key, cls in SEG_QUESTIONS.items():
                idx = ICON_CLASSES.index(cls)
                boxes[key] = boxes_from_mask((icon_map == idx) & (icon_p >= 0.35), ICON_MIN_AREA[cls], scale)
            boxes["bedrooms"] = boxes_from_mask(room_map == ROOM_CLASSES.index("Bed Room"), BED_MIN_AREA, scale)
            rec["boxes"] = boxes
            fp.write(json.dumps(rec) + "\n")
            fp.flush()
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  {i + 1}/{len(todo)} {Path(rel).name} {time.time() - t:.1f}s  elapsed {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(todo)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
