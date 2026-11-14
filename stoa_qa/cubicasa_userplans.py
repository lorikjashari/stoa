"""Teach the raster detector the user's own floor plans: label I/O, splits, data mixing, threshold tuning, scoring.

Pure logic shared by the local job (modal_jobs/cubicasa_userplans.py), the Modal job
(modal_jobs/cubicasa_userplans_cloud.py) and the offline tests. Stdlib only: it is imported inside Modal containers.

A user plan folder looks like this (the jobs create everything but images/):

    myplans/
      images/<plan>.png|jpg     the user's plans (or the image files directly in myplans/)
      labels_auto/<plan>.txt    pseudo-labels from the current detector (YOLO format, 13 classes) - NOT ground truth
      labels/<plan>.txt         corrected labels (saved by tools/plan_labeler.html) - the only labels used to train/score
      splits.json               which corrected plan is train / val / test; a plan never changes split once assigned
      predictions/<run>.json    low-confidence boxes of a detector run for every plan, [cls, score, x0, y0, x1, y1]

Honesty rule, as for CubiCasa: train on the user's train plans (+ a CubiCasa train replay sample), choose thresholds on
val (user val + CubiCasa val), report only on the held-out test plans (user test + CubiCasa test). Never tune on test.
"""
from __future__ import annotations

import collections
import json
import os
import random
import shutil
from pathlib import Path

from .cubicasa_rules import CLASSES, QUESTION_CLASSES, select_boxes

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
# The same grid stoa_qa.cubicasa_hybrid searches (kept equal by tests/test_userplans.py).
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 19)]
IOUS = [0.3, 0.5, 0.7, 1.0]
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------- label I/O (YOLO txt, 13 classes)

def to_yolo_lines(boxes: list, w: int, h: int) -> list[str]:
    """Boxes [cls, score, x0, y0, x1, y1] (or [cls, x0, y0, x1, y1]) in pixels -> YOLO lines 'cls cx cy bw bh' (0..1).

    Boxes are clipped to the image; slivers under 2 px are dropped (as in cubicasa_local.labels)."""
    lines = []
    for b in boxes:
        c, (x0, y0, x1, y1) = int(b[0]), b[-4:]
        if not 0 <= c < len(CLASSES):
            raise ValueError(f"class id {c} outside 0..{len(CLASSES) - 1}")
        x0, x1 = max(0.0, min(x0, x1)), min(float(w), max(x0, x1))
        y0, y1 = max(0.0, min(y0, y1)), min(float(h), max(y0, y1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        lines.append(f"{c} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return lines


def from_yolo_text(text: str, w: int, h: int) -> list[list]:
    """YOLO label text -> boxes [cls, 1.0, x0, y0, x1, y1] in pixels. Raises ValueError naming the bad line."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        p = line.split()
        try:
            c = int(p[0])
            cx, cy, bw, bh = (float(v) for v in p[1:5])
        except (ValueError, IndexError):
            raise ValueError(f"line {i}: expected 'class cx cy w h', got {line!r}") from None
        if len(p) != 5 or not 0 <= c < len(CLASSES):
            raise ValueError(f"line {i}: expected 'class cx cy w h' with class 0..{len(CLASSES) - 1}, got {line!r}")
        out.append([c, 1.0, round((cx - bw / 2) * w, 1), round((cy - bh / 2) * h, 1),
                    round((cx + bw / 2) * w, 1), round((cy + bh / 2) * h, 1)])
    return out


def write_labels(path: Path, boxes: list, w: int, h: int) -> int:
    lines = to_yolo_lines(boxes, w, h)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def label_counts(text: str) -> dict[str, int]:
    """Per-question answer implied by a corrected label file: every labelled box counts (as Stoa counts annotations)."""
    per_class = collections.Counter(int(line.split()[0]) for line in text.splitlines() if line.strip())
    return {q: sum(per_class[CLASSES.index(c)] for c in cs) for q, cs in QUESTION_CLASSES.items()}


def classes_txt() -> str:
    """classes.txt as LabelImg / Label Studio's YOLO import expect it: one name per line, line number = class id."""
    return "\n".join(CLASSES) + "\n"


# ---------------------------------------------------------------- pseudo-labels

def load_picks(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    p = json.loads(Path(path).read_text())
    return p["yolo_thresholds"], p.get("yolo_iou", {})


def pseudo_label(boxes: list, thr: dict[str, float], iou: dict[str, float] | None = None) -> list:
    """The boxes the app would count with these per-question thresholds/overlap cuts (each class belongs to one question)."""
    out = []
    for q in QUESTION_CLASSES:
        out += select_boxes(boxes, q, thr[q], (iou or {}).get(q, 1.0))
    return out


# ---------------------------------------------------------------- splits of the user's plans

def assign_splits(names, existing: dict[str, str] | None = None, val_frac: float = 0.2, test_frac: float = 0.2,
                  seed: int = 0) -> dict[str, str]:
    """name -> train/val/test. Plans keep the split they already have (a test plan never moves into training when more
    plans are added); new plans fill val/test up to their share of the total first, the rest go to train.
    With 3+ plans, val and test get at least one plan each."""
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1:
        raise ValueError("need val_frac, test_frac >= 0 and val_frac + test_frac < 1")
    names = sorted(set(names))
    out = {n: s for n, s in (existing or {}).items() if n in names and s in SPLITS}
    new = [n for n in names if n not in out]
    random.Random(seed).shuffle(new)
    total = len(names)
    want = {"test": round(total * test_frac), "val": round(total * val_frac)}
    if total >= 3:
        want = {s: max(1, k) if f > 0 else k for (s, k), f in zip(want.items(), (test_frac, val_frac))}
    for n in new:
        have = collections.Counter(out.values())
        out[n] = "test" if have["test"] < want["test"] else "val" if have["val"] < want["val"] else "train"
    return dict(sorted(out.items()))


# ---------------------------------------------------------------- the fine-tuning dataset

def mix_plan(user_splits: dict[str, str], cubicasa: dict[str, list[str]], repeat: int = 3, replay_ratio: float = 2.0,
             replay_val: int = 100, seed: int = 0) -> list[tuple[str, str, str, str]]:
    """Entries (source, plan, dest_split, dest_name) of the fine-tuning dataset.

    train = each user train plan `repeat` times + a random replay sample of CubiCasa TRAIN plans
            (`replay_ratio` CubiCasa images per user training image) so the model keeps what it learned;
    val   = user val plans + up to `replay_val` CubiCasa VAL plans (ultralytics picks best.pt on this mix);
    test  = user test plans only. CubiCasa test plans never enter this dataset (they are scored separately)."""
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    entries = []
    for name, split in sorted(user_splits.items()):
        if split == "train":
            entries += [("user", name, "train", f"u_{name}" + (f"__r{r}" if r else "")) for r in range(repeat)]
        elif split in ("val", "test"):
            entries.append(("user", name, split, f"u_{name}"))
    rng = random.Random(seed)
    n_user_train = sum(e[2] == "train" for e in entries)
    for split, k in (("train", round(replay_ratio * n_user_train)), ("val", replay_val)):
        pool = sorted(cubicasa.get(split, []))
        entries += [("cubicasa", n, split, f"cc_{n}") for n in sorted(rng.sample(pool, min(k, len(pool))))]
    return entries


def images_dir(root: Path) -> Path:
    """root/images when it exists, else root itself (a plain folder of plan images works too)."""
    return root / "images" if (root / "images").is_dir() else root


def list_images(root: Path) -> dict[str, Path]:
    """plan name (file stem) -> image path."""
    return {p.stem: p for p in sorted(images_dir(root).iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS}


def _link(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)  # same disk: no extra space
    except OSError:
        shutil.copy2(src, dest)


def materialize(entries, user_dir: Path, cubicasa_dir: Path | None, out: Path) -> Path:
    """Lay the dataset out as ultralytics expects (images/<split>/x.ext + labels/<split>/x.txt) and write data.yaml.

    User plans come from user_dir(/images) + user_dir/labels (corrected labels only); CubiCasa plans from
    cubicasa_dir/images/<split> + labels/<split> (the layout cubicasa_local.py / cubicasa_detector.py write)."""
    if out.exists():
        shutil.rmtree(out)  # stale files from an earlier mix must not leak into this one
    user_imgs = list_images(user_dir)
    for source, name, split, dest in entries:
        if source == "user":
            img, lbl = user_imgs[name], user_dir / "labels" / f"{name}.txt"
        else:
            img = cubicasa_dir / "images" / split / f"{name}.png"
            lbl = cubicasa_dir / "labels" / split / f"{name}.txt"
        _link(img, out / "images" / split / f"{dest}{img.suffix.lower()}")
        _link(lbl, out / "labels" / split / f"{dest}.txt")
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    (out / "manifest.json").write_text(json.dumps([list(e) for e in entries], indent=0))
    return out / "data.yaml"


def finetune_args(epochs: int = 15, lr0: float = 2e-4, freeze: int = 10, imgsz: int = 1280, batch: int = 2,
                  optimizer: str = "AdamW", patience: int = 8) -> dict:
    """ultralytics train() arguments for a gentle fine-tune from an existing best.pt.

    The optimizer is set explicitly because ultralytics' 'auto' ignores lr0; freezing the first 10 layers (the YOLO11
    backbone) and a low learning rate keep what was learned on CubiCasa. Flips as in the original training."""
    return dict(epochs=epochs, lr0=lr0, lrf=0.1, optimizer=optimizer, freeze=freeze or None, imgsz=imgsz, batch=batch,
                warmup_epochs=min(1.0, epochs / 10), close_mosaic=min(3, epochs), patience=patience,
                flipud=0.5, fliplr=0.5, plots=False)


# ---------------------------------------------------------------- scoring and val-only threshold tuning

def count_accuracy(preds: dict[str, list], truth: dict[str, dict[str, int]], thr: dict[str, float],
                   iou: dict[str, float] | None = None, plans=None) -> dict[str, tuple[int, int]]:
    """question -> (exact counts, questions scored) over `plans` (default: every plan with both preds and truth)."""
    out = {}
    plans = [p for p in (plans if plans is not None else truth) if p in preds and p in truth]
    for q in QUESTION_CLASSES:
        ok = n = 0
        for p in plans:
            if q in truth[p]:
                n += 1
                ok += len(select_boxes(preds[p], q, thr[q], (iou or {}).get(q, 1.0))) == truth[p][q]
        out[q] = (ok, n)
    return out


def total(acc: dict[str, tuple[int, int]]) -> tuple[int, int]:
    return sum(ok for ok, _ in acc.values()), sum(n for _, n in acc.values())


def tune(val_sets: list[tuple[dict, dict, float]]) -> tuple[dict[str, float], dict[str, float]]:
    """Per question, the (threshold, overlap cut) with the best weighted exact-count rate over VAL sets only.

    val_sets: (preds, truth, weight) - e.g. the user's val plans and CubiCasa val, weighted so a handful of user plans is
    not drowned by 400 CubiCasa plans. Ties go to the simpler setting, as in cubicasa_hybrid.tune_yolo."""
    thr, iou = {}, {}
    for q in QUESTION_CLASSES:
        sets = [([(preds[p], truth[p][q]) for p in truth if p in preds and q in truth[p]], w) for preds, truth, w in val_sets]
        sets = [(items, w) for items, w in sets if items]

        def score(tu):
            return sum(w * sum(len(select_boxes(b, q, tu[0], tu[1])) == g for b, g in items) / len(items) for items, w in sets)

        thr[q], iou[q] = max(((t, u) for t in THRESHOLDS for u in IOUS), key=lambda tu: (score(tu), tu[1], -abs(tu[0] - 0.5)))
    return thr, iou
