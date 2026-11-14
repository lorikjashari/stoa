"""Teach the raster detector YOUR floor plans (local GPU): pseudo-label -> correct -> split -> mix -> fine-tune -> evaluate.

The detector was trained on Finnish CubiCasa5K plans. Plans drawn in another style (other symbols, colours, scans) are
counted worse. This job starts from the trained best.pt and fine-tunes it on your corrected plans mixed with a replay
sample of CubiCasa train plans, so it learns your style without forgetting CubiCasa.

    PY=../.venv-train/Scripts/python                     # any env with ultralytics (+ CUDA torch for a GPU)
    $PY modal_jobs/cubicasa_userplans.py label   myplans                 # boxes from the current detector -> labels_auto/
    #   open tools/plan_labeler.html in Chrome/Edge, pick myplans/, fix boxes, save -> labels/  (only these count)
    $PY modal_jobs/cubicasa_userplans.py split   myplans                 # train/val/test of the corrected plans (sticky)
    $PY modal_jobs/cubicasa_userplans.py prepare myplans --name ft-mine  # user train x3 + CubiCasa replay -> data.yaml
    $PY modal_jobs/cubicasa_userplans.py train   --name ft-mine          # fine-tune from best.pt, low lr, frozen backbone
    $PY modal_jobs/cubicasa_userplans.py predict myplans --tag ft-mine --weights eval_out/yolo_runs/ft-mine/weights/best.pt
    $PY modal_jobs/cubicasa_userplans.py evaluate myplans --before m13-1280 --after ft-mine
    $PY modal_jobs/cubicasa_userplans.py demo                            # the whole loop on synthetic plans (dry run)

Base weights: --weights, else third_party/weights/m13-1280.pt (or m13-1280/best.pt), else eval_out/yolo_runs/m13-1280/weights/best.pt.
Get them with  modal volume get stoa-cubicasa runs/m13-1280/weights/best.pt third_party/weights/m13-1280.pt
No local GPU? modal_jobs/cubicasa_userplans_cloud.py does label + fine-tune + predict on Modal with the same files.

Honesty rule: train on your train plans (+ CubiCasa train), pick thresholds on val (yours + CubiCasa val), report on the
held-out test plans (yours + CubiCasa test). Pseudo-labels are never used as truth; only corrected labels/ are.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from stoa_qa import cubicasa_userplans as U  # noqa: E402
from stoa_qa.cubicasa_rules import CLASSES, QUESTION_CLASSES  # noqa: E402

PICKS = REPO / "stoa_qa" / "cubicasa_picks.json"
WEIGHT_CANDIDATES = [REPO / "third_party/weights/m13-1280.pt", REPO / "third_party/weights/m13-1280/best.pt",
                     REPO / "eval_out/yolo_runs/m13-1280/weights/best.pt"]


def _device(d: str | None) -> str:
    if d:
        return d
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _weights(given: list[str] | None) -> list[Path]:
    if given:
        paths = [Path(w) for w in given]
        for p in paths:
            if not p.exists():
                raise SystemExit(f"{p} not found")
        return paths
    for p in WEIGHT_CANDIDATES:
        if p.exists():
            return [p]
    raise SystemExit("no detector weights found. Get the trained model first:\n"
                     "  modal volume get stoa-cubicasa runs/m13-1280/weights/best.pt third_party/weights/m13-1280.pt\n"
                     "or pass --weights path/to/best.pt (or run the labelling on Modal: modal_jobs/cubicasa_userplans_cloud.py)")


def _tag(weights: list[Path]) -> str:
    """runs/<name>/weights/best.pt or weights/<name>/best.pt -> <name>; other.pt -> other; two weights -> ens-<a>-<b>."""
    def name(w: Path) -> str:
        if w.stem not in ("best", "last"):
            return w.stem  # third_party/weights/m13-1280.pt
        return w.parent.parent.name if w.parent.name == "weights" else w.parent.name

    names = [name(w) for w in weights]
    return names[0] if len(names) == 1 else "ens-" + "-".join(names)


def predict_boxes(weights: list[Path], files: list[Path], imgsz: int, device: str, conf: float = 0.05, batch: int = 4) -> dict:
    """{plan: [[cls, score, x0, y0, x1, y1], ...]} in original image pixels at low confidence (thresholds are applied later).
    Two weights are fused as in stoa_qa.cubicasa_ensemble (the app's current run ens-m13-s13 is such a fusion)."""
    from ultralytics import YOLO

    from stoa_qa.cubicasa_ensemble import fuse

    runs = []
    for w in weights:
        m = YOLO(str(w))
        out = {}
        for i in range(0, len(files), batch):
            chunk = [str(f) for f in files[i:i + batch]]
            for f, r in zip(chunk, m.predict(chunk, imgsz=imgsz, conf=conf, verbose=False, device=device)):
                b = r.boxes
                out[Path(f).stem] = [[int(c), round(float(s), 4), *[round(float(v), 1) for v in xyxy]]
                                     for c, s, xyxy in zip(b.cls, b.conf, b.xyxy)]
        runs.append(out)
    if len(runs) == 1:
        return runs[0]
    if len(runs) != 2:
        raise SystemExit("give one or two --weights")
    return {k: fuse(runs[0][k], runs[1].get(k, [])) for k in runs[0]}


def _size(path: Path) -> tuple[int, int]:
    """Displayed size (EXIF rotation applied, as browsers and ultralytics do)."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).size


def write_pseudo_labels(root: Path, preds: dict, thr: dict, iou: dict, overwrite: bool = False) -> tuple[int, int]:
    """labels_auto/<plan>.txt = the boxes the app would count; never touches corrected labels/."""
    imgs = U.list_images(root)
    n = boxes = 0
    for name, b in preds.items():
        dest = root / "labels_auto" / f"{name}.txt"
        if dest.exists() and not overwrite:
            continue
        w, h = _size(imgs[name])
        boxes += U.write_labels(dest, U.pseudo_label(b, thr, iou), w, h)
        n += 1
    (root / "classes.txt").write_text(U.classes_txt())
    return n, boxes


def save_preds(root: Path, tag: str, preds: dict) -> Path:
    f = root / "predictions" / f"{tag}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(preds))
    return f


# ---------------------------------------------------------------- commands

def cmd_label(root: Path, weights: list[str] | None, picks: Path, imgsz: int, device: str | None, overwrite: bool,
              tag: str | None = None) -> None:
    imgs = U.list_images(root)
    if not imgs:
        raise SystemExit(f"no plan images ({', '.join(U.IMAGE_EXTS)}) in {U.images_dir(root)}")
    w = _weights(weights)
    tag = tag or _tag(w)
    thr, iou = U.load_picks(picks)
    picks_run = json.loads(picks.read_text()).get("yolo_run")
    if picks_run != tag:
        print(f"note: thresholds in {picks.name} were tuned on val for run '{picks_run}', applied here to '{tag}'. "
              "Fine for pseudo-labels you correct by hand; the evaluation re-tunes them on val.")
    t = time.time()
    preds = predict_boxes(w, list(imgs.values()), imgsz, _device(device))
    f = save_preds(root, tag, preds)
    n, boxes = write_pseudo_labels(root, preds, thr, iou, overwrite)
    print(f"{len(preds)} plans in {time.time() - t:.0f}s: raw boxes -> {f}; {n} pseudo-label files ({boxes} boxes) -> "
          f"{root / 'labels_auto'}\nnext: open tools/plan_labeler.html, choose {root}, correct each plan and save "
          "(writes labels/<plan>.txt). Only corrected plans are used.")


def cmd_split(root: Path, val: float, test: float, seed: int) -> dict:
    corrected = sorted(p.stem for p in (root / "labels").glob("*.txt")) if (root / "labels").is_dir() else []
    imgs = U.list_images(root)
    missing = [n for n in corrected if n not in imgs]
    if missing:
        raise SystemExit(f"labels without an image: {missing[:5]}")
    f = root / "splits.json"
    old = json.loads(f.read_text())["plans"] if f.exists() else {}
    splits = U.assign_splits(corrected, old, val, test, seed)
    moved = [n for n in old if n in splits and splits[n] != old[n]]
    assert not moved, moved  # assign_splits never moves a plan
    f.write_text(json.dumps({"seed": seed, "val_frac": val, "test_frac": test, "plans": splits}, indent=1))
    c = {s: sum(v == s for v in splits.values()) for s in U.SPLITS}
    unlabelled = len(imgs) - len(corrected)
    print(f"{len(splits)} corrected plans -> {c} ({f}); {unlabelled} plans without corrected labels are not used")
    if c["test"] < 5:
        print(f"warning: only {c['test']} test plans - the before/after numbers will be very noisy. Correct more plans.")
    return splits


def cmd_prepare(root: Path, name: str, cubicasa: Path, eval_out: Path, repeat: int, replay_ratio: float, replay_val: int,
                seed: int) -> Path:
    f = root / "splits.json"
    if not f.exists():
        raise SystemExit(f"{f} missing: run  split {root}  first")
    splits = json.loads(f.read_text())["plans"]
    cc = {s: sorted(p.stem for p in (cubicasa / "labels" / s).glob("*.txt")) for s in ("train", "val")}
    if not cc["train"] and replay_ratio > 0:
        print(f"warning: no CubiCasa labels under {cubicasa}/labels/train - training WITHOUT replay; the model may forget "
              "CubiCasa. Build them with cubicasa_local.py extract+labels, or fine-tune on Modal where they exist.")
    entries = U.mix_plan(splits, cc, repeat, replay_ratio, replay_val, seed)
    out = eval_out / "userplans" / name / "data"
    yaml = U.materialize(entries, root, cubicasa, out)
    c = {(src, s): 0 for src in ("user", "cubicasa") for s in U.SPLITS}
    for src, _, s, _ in entries:
        c[(src, s)] += 1
    print("dataset:", {f"{s}/{src}": v for (src, s), v in c.items() if v}, "->", yaml)
    return yaml


def cmd_train(name: str, eval_out: Path, weights: list[str] | None, device: str | None, workers: int, amp: bool = True,
              **hp) -> Path:
    from ultralytics import YOLO

    data = eval_out / "userplans" / name / "data" / "data.yaml"
    if not data.exists():
        raise SystemExit(f"{data} missing: run  prepare <plans> --name {name}  first")
    base = _weights(weights)[0]
    args = U.finetune_args(**hp)
    print(f"fine-tuning {base} -> {eval_out / 'yolo_runs' / name}: {args}")
    m = YOLO(str(base))
    m.train(data=str(data), project=str(eval_out / "yolo_runs"), name=name, exist_ok=True, device=_device(device),
            workers=workers, amp=amp, **args)
    best = eval_out / "yolo_runs" / name / "weights" / "best.pt"
    print(f"best.pt (chosen on the val mix) -> {best}\nnext: predict <plans> --tag {name} --weights {best}")
    return best


def cmd_predict(root: Path, weights: list[str] | None, tag: str | None, imgsz: int, device: str | None, cubicasa: Path,
                eval_out: Path, skip_cubicasa: bool) -> str:
    """Raw low-confidence boxes for every user plan and (when the images are here) CubiCasa val + test."""
    w = _weights(weights)
    tag = tag or _tag(w)
    dev = _device(device)
    imgs = U.list_images(root)
    print(f"{len(imgs)} user plans -> {save_preds(root, tag, predict_boxes(w, list(imgs.values()), imgsz, dev))}")
    if skip_cubicasa:
        return tag
    dest = eval_out / "yolo" / tag
    for split in ("val", "test"):
        files = sorted((cubicasa / "images" / split).glob("*.png"))
        if not files:
            print(f"no CubiCasa {split} images under {cubicasa} - skipped (the Modal job predicts them on the Volume)")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{split}.json").write_text(json.dumps(predict_boxes(w, files, imgsz, dev)))
        print(f"{len(files)} CubiCasa {split} plans -> {dest / (split + '.json')}")
    return tag


def _cubicasa_truth(cubicasa: Path, eval_out: Path, tags: list[str]) -> tuple[dict, str]:
    """{plan stem: {question: answer}} for CubiCasa val+test, and where it came from (best source first)."""
    try:
        from stoa_qa.cubicasa_gemini import load_truth

        truth = {Path(k).stem: v for k, v in load_truth().items()}
        if truth:
            return truth, "Stoa answers (questions.jsonl): the same metric as the published 84.2%"
    except (ImportError, OSError):
        pass
    truth = {}
    for split in ("val", "test"):
        for f in sorted((cubicasa / "labels" / split).glob("*.txt")):
            truth[f.stem] = U.label_counts(f.read_text())
    if truth:
        return truth, f"label counts from {cubicasa}/labels (CubiCasa annotations; match Stoa's answers on 98-100% of plans)"
    for t in tags:
        f = eval_out / "yolo" / t / "label_truth.json"
        if f.exists():
            for split in json.loads(f.read_text()).values():
                truth.update(split)
            return truth, f"label counts saved by the Modal job ({f})"
    return {}, "none (no dataset, no CubiCasa labels)"


def _cc_preds(eval_out: Path, tag: str, split: str) -> dict:
    f = eval_out / "yolo" / tag / f"{split}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _fmt(acc: dict) -> str:
    ok, n = U.total(acc)
    return f"{ok / n:6.1%} ({ok}/{n})" if n else "     -"


def cmd_evaluate(root: Path, before: str, after: str, picks: Path, cubicasa: Path, eval_out: Path, user_weight: float) -> dict:
    splits = json.loads((root / "splits.json").read_text())["plans"]
    truth_u = {n: U.label_counts((root / "labels" / f"{n}.txt").read_text()) for n in splits}
    pv = {s: [n for n, v in splits.items() if v == s] for s in U.SPLITS}
    preds_u = {t: json.loads((root / "predictions" / f"{t}.json").read_text()) for t in (before, after)}
    cc_truth, cc_src = _cubicasa_truth(cubicasa, eval_out, [after, before])
    cc = {t: {s: _cc_preds(eval_out, t, s) for s in ("val", "test")} for t in (before, after)}
    base_thr, base_iou = U.load_picks(picks)
    base_meta = json.loads(picks.read_text())

    def val_sets(t):
        sets = [(preds_u[t], {n: truth_u[n] for n in pv["val"]}, user_weight)]
        if cc[t]["val"] and cc_truth:
            sets.append((cc[t]["val"], cc_truth, 1 - user_weight))
        return sets

    rows = []
    for label, t, thr_iou in (
            (f"before: {before}, thresholds from {picks.name} (run {base_meta.get('yolo_run')})", before, (base_thr, base_iou)),
            (f"before: {before}, thresholds re-tuned on val", before, U.tune(val_sets(before))),
            (f"after:  {after}, thresholds tuned on val", after, U.tune(val_sets(after)))):
        thr, iou = thr_iou
        rows.append({"row": label, "run": t, "thresholds": thr, "iou": iou,
                     "user_test": U.count_accuracy(preds_u[t], truth_u, thr, iou, pv["test"]),
                     "user_test_nonzero": U.count_accuracy(
                         preds_u[t], {n: {q: v for q, v in truth_u[n].items() if v} for n in pv["test"]}, thr, iou),
                     "cubicasa_test": U.count_accuracy(cc[t]["test"], cc_truth, thr, iou) if cc[t]["test"] else {}})

    print(f"\nyour plans: {len(pv['train'])} train / {len(pv['val'])} val / {len(pv['test'])} test (scored: test only)")
    print(f"CubiCasa truth: {cc_src}")
    print(f"thresholds tuned on: your val plans (weight {user_weight}) + CubiCasa val (weight {1 - user_weight:.2f}); never on test\n")
    print(f"{'':62}{'your test':>18}{'(nonzero answers)':>20}{'CubiCasa test':>22}")
    for r in rows:
        print(f"{r["row"][:72]:72}{_fmt(r['user_test']):>18}{_fmt(r['user_test_nonzero']):>20}{_fmt(r['cubicasa_test']):>22}")
    if not cc[before]["test"]:
        print(f"  (no CubiCasa test boxes for '{before}': published reference = {base_meta.get('test_accuracy', 0):.1%} "
              f"on {base_meta.get('test_questions')} questions for run {base_meta.get('yolo_run')})")
    print(f"\n{'per question (your test)':<26}" + "".join(f"{q[:9]:>10}" for q in QUESTION_CLASSES))
    for r in rows:
        print(f"{r['run'][:24]:<26}" + "".join(
            f"{(f'{ok}/{n}' if n else '-'):>10}" for ok, n in r["user_test"].values()))
    if len(pv["test"]) < 5:
        print(f"\nwarning: {len(pv['test'])} test plans is too few to trust a difference; correct more plans and re-run split.")

    a = rows[-1]
    out = eval_out / "userplans" / after
    out.mkdir(parents=True, exist_ok=True)
    pk = {"selected_on": f"user val ({len(pv['val'])} plans, weight {user_weight}) + CubiCasa val",
          "yolo_run": after, "yolo_thresholds": a["thresholds"], "yolo_iou": a["iou"],
          "user_test_accuracy": round(U.total(a["user_test"])[0] / max(1, U.total(a["user_test"])[1]), 4),
          "user_test_questions": U.total(a["user_test"])[1],
          "cubicasa_test_accuracy": round(U.total(a["cubicasa_test"])[0] / max(1, U.total(a["cubicasa_test"])[1]), 4)
          if a["cubicasa_test"] else None,
          "cubicasa_truth": cc_src}
    (out / "picks.json").write_text(json.dumps(pk, indent=1) + "\n")
    (out / "report.json").write_text(json.dumps({"rows": rows, "splits": splits, "cubicasa_truth": cc_src}, indent=1))
    print(f"\nthresholds for '{after}' (same format as stoa_qa/cubicasa_picks.json) -> {out / 'picks.json'}")
    return {"rows": rows, "picks": pk}


# ---------------------------------------------------------------- dry run on synthetic plans

def _synth_plan(rng: random.Random, style: str, size: int = 256):
    """A toy 'plan': outer wall (floor), door arcs, wall windows, toilet ellipses. Two drawing styles = domain shift."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(im)
    ink, lw = ((0, 0, 0), 2) if style == "cubicasa" else ((40, 70, 200), 4)
    m = rng.randint(10, 24)
    d.rectangle([m, m, size - m, size - m], outline=ink, width=lw)
    boxes = [[CLASSES.index("floor"), m, m, size - m, size - m]]
    for _ in range(rng.randint(1, 4)):
        x = rng.randint(m + 20, size - m - 60)
        d.rectangle([x, m - 4, x + 34, m + 4], outline=ink, width=1 if style == "cubicasa" else 3)
        boxes.append([CLASSES.index("window"), x, m - 4, x + 34, m + 4])
    for _ in range(rng.randint(1, 3)):
        x, y, r = rng.randint(m + 30, size - m - 50), rng.randint(m + 30, size - m - 50), rng.randint(16, 24)
        d.arc([x - r, y - r, x + r, y + r], 0, 90, fill=ink, width=lw)
        d.line([x, y, x + r, y], fill=ink, width=lw)
        boxes.append([CLASSES.index("door"), x, y, x + r, y + r])
    if rng.random() < 0.7:
        x, y = rng.randint(m + 20, size - m - 30), rng.randint(m + 20, size - m - 40)
        d.ellipse([x, y, x + 14, y + 20], outline=ink, width=lw)
        boxes.append([CLASSES.index("toilet"), x, y, x + 14, y + 20])
    return im, boxes


def cmd_demo(out: Path, device: str | None, epochs: int, train_plans: int = 24) -> None:
    """End to end on synthetic plans (no dataset, no downloads): base model -> label -> 'correct' -> split -> prepare ->
    fine-tune -> predict -> evaluate. The numbers are meaningless; it proves the plumbing runs."""
    from ultralytics import YOLO

    rng = random.Random(0)
    size = 256
    cc, plans = out / "cubicasa", out / "myplans"
    for split, n in (("train", train_plans), ("val", 8), ("test", 8)):
        for i in range(n):
            im, b = _synth_plan(rng, "cubicasa", size)
            (cc / "images" / split).mkdir(parents=True, exist_ok=True)
            im.save(cc / "images" / split / f"cc{split}{i:02d}.png")
            U.write_labels(cc / "labels" / split / f"cc{split}{i:02d}.txt", b, size, size)
    truth = {}
    (plans / "images").mkdir(parents=True, exist_ok=True)
    for i in range(12):
        im, b = _synth_plan(rng, "user", size)
        im.save(plans / "images" / f"plan{i:02d}.png")
        truth[f"plan{i:02d}"] = b
    eo = out / "eval_out"
    base_yaml = U.materialize([("cubicasa", p.stem, s, p.stem) for s in U.SPLITS
                               for p in sorted((cc / "images" / s).glob("*.png"))], plans, cc, out / "base_data")
    print("== 1. base model on synthetic 'CubiCasa' plans (stands in for m13-1280)")
    YOLO("yolo11n.yaml").train(data=str(base_yaml), epochs=epochs, imgsz=size, batch=8, workers=0, device=_device(device),
                               project=str(eo / "yolo_runs"), name="demo-base", exist_ok=True, plots=False, amp=False,
                               verbose=False)
    base = str(eo / "yolo_runs/demo-base/weights/best.pt")
    print("== 2. pseudo-label the user's plans")
    cmd_label(plans, [base], PICKS, size, device, overwrite=True)
    print("== 3. 'correct' them (the demo writes the synthetic ground truth to labels/, as the labeler would)")
    for n, b in truth.items():
        U.write_labels(plans / "labels" / f"{n}.txt", b, size, size)
    print("== 4. split")
    cmd_split(plans, 0.25, 0.25, 0)
    print("== 5. prepare (user train x3 + CubiCasa replay)")
    cmd_prepare(plans, "demo-ft", cc, eo, repeat=3, replay_ratio=1.0, replay_val=4, seed=0)
    print("== 6. fine-tune")
    best = cmd_train("demo-ft", eo, [base], device, workers=0, amp=False, epochs=epochs, imgsz=size, batch=8, freeze=10)
    print("== 7. predict (user plans + CubiCasa val/test) with both models")
    cmd_predict(plans, [base], "demo-base", size, device, cc, eo, False)
    cmd_predict(plans, [str(best)], "demo-ft", size, device, cc, eo, False)
    print("== 8. evaluate (numbers are meaningless on toy data)")
    cmd_evaluate(plans, "demo-base", "demo-ft", PICKS, cc, eo, 0.5)
    print(f"\ndry run OK -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-out", type=Path, default=REPO / "eval_out", help="where runs, datasets and reports go")
    ap.add_argument("--cubicasa", type=Path, default=REPO / "eval_out/yolo_data",
                    help="CubiCasa YOLO dataset (images/<split>, labels/<split>) from cubicasa_local.py labels")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def plans_arg(p):
        p.add_argument("plans", type=Path, help="folder of your plans (images/ inside, or the images directly)")

    p = sub.add_parser("label", help="pseudo-label your plans with the current detector -> labels_auto/")
    plans_arg(p)
    p.add_argument("--weights", nargs="+", help="one best.pt, or two to fuse like ens-m13-s13")
    p.add_argument("--tag", help="name for these predictions (default: the run name of the weights)")
    p.add_argument("--picks", type=Path, default=PICKS)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device")
    p.add_argument("--overwrite", action="store_true", help="redo existing labels_auto files")

    p = sub.add_parser("split", help="assign corrected plans to train/val/test (a plan never changes split)")
    plans_arg(p)
    p.add_argument("--val", type=float, default=0.2)
    p.add_argument("--test", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("prepare", help="mix your train plans with a CubiCasa replay sample -> data.yaml")
    plans_arg(p)
    p.add_argument("--name", required=True)
    p.add_argument("--repeat", type=int, default=3, help="copies of each of your train plans per epoch")
    p.add_argument("--replay-ratio", type=float, default=2.0, help="CubiCasa train plans per user training image")
    p.add_argument("--replay-val", type=int, default=100, help="CubiCasa val plans in the val mix")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("train", help="fine-tune from best.pt on the prepared mix")
    p.add_argument("--name", required=True)
    p.add_argument("--weights", nargs=1, help="base best.pt (default: m13-1280)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr0", type=float, default=2e-4)
    p.add_argument("--freeze", type=int, default=10, help="freeze the first N layers (10 = YOLO11 backbone; 0 = none)")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=2, help="2 fits YOLO11m at 1280 px on a 6 GB laptop GPU")
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device")

    p = sub.add_parser("predict", help="low-confidence boxes of a model for your plans (+ CubiCasa val/test)")
    plans_arg(p)
    p.add_argument("--weights", nargs="+")
    p.add_argument("--tag")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device")
    p.add_argument("--skip-cubicasa", action="store_true")

    p = sub.add_parser("evaluate", help="before/after counts accuracy on held-out test plans; thresholds tuned on val")
    plans_arg(p)
    p.add_argument("--before", required=True, help="predictions tag of the base model")
    p.add_argument("--after", required=True, help="predictions tag of the fine-tuned model")
    p.add_argument("--picks", type=Path, default=PICKS)
    p.add_argument("--user-weight", type=float, default=0.5, help="weight of your val plans vs CubiCasa val when tuning")

    p = sub.add_parser("demo", help="dry run of the whole loop on synthetic plans")
    p.add_argument("--out", type=Path, default=REPO / "eval_out/userplans_demo")
    p.add_argument("--device")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--train-plans", type=int, default=24, help="synthetic CubiCasa-style train plans for the toy base model")

    a = ap.parse_args()
    if a.cmd == "label":
        cmd_label(a.plans, a.weights, a.picks, a.imgsz, a.device, a.overwrite, a.tag)
    elif a.cmd == "split":
        cmd_split(a.plans, a.val, a.test, a.seed)
    elif a.cmd == "prepare":
        cmd_prepare(a.plans, a.name, a.cubicasa, a.eval_out, a.repeat, a.replay_ratio, a.replay_val, a.seed)
    elif a.cmd == "train":
        cmd_train(a.name, a.eval_out, a.weights, a.device, a.workers, epochs=a.epochs, lr0=a.lr0, freeze=a.freeze,
                  imgsz=a.imgsz, batch=a.batch, optimizer=a.optimizer, patience=a.patience)
    elif a.cmd == "predict":
        cmd_predict(a.plans, a.weights, a.tag, a.imgsz, a.device, a.cubicasa, a.eval_out, a.skip_cubicasa)
    elif a.cmd == "evaluate":
        cmd_evaluate(a.plans, a.before, a.after, a.picks, a.cubicasa, a.eval_out, a.user_weight)
    elif a.cmd == "demo":
        if a.out.exists():
            import shutil

            shutil.rmtree(a.out)
        cmd_demo(a.out, a.device, a.epochs, a.train_plans)


if __name__ == "__main__":
    main()
