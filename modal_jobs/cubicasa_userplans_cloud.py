"""Teach the raster detector YOUR floor plans on Modal (no local GPU or weights needed). Same steps and files as
modal_jobs/cubicasa_userplans.py; the CubiCasa data and the trained runs are on the Volume `stoa-cubicasa`
(built by cubicasa_detector.py / cubicasa_cloud.py).

    modal run modal_jobs/cubicasa_userplans_cloud.py::label --plans myplans                 # -> myplans/labels_auto/
    #   correct them in tools/plan_labeler.html (saves myplans/labels/), then, locally (no GPU needed):
    python modal_jobs/cubicasa_userplans.py split myplans
    modal run --detach modal_jobs/cubicasa_userplans_cloud.py::finetune --plans myplans --name ft-mine
    python modal_jobs/cubicasa_userplans.py evaluate myplans --before m13-1280 --after ft-mine

`finetune` uploads your corrected plans, mixes them with a CubiCasa train replay sample, fine-tunes runs/<base>/weights/
best.pt into runs/<name>/ on the Volume, then predicts your plans and CubiCasa val + test with both the base and the new
model and writes the boxes locally (myplans/predictions/, eval_out/yolo/<run>/), ready for `evaluate`.
Your plan images are sent to Modal for this; only the weights stay on the Volume.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
VOL = "/vol"
RUNS = f"{VOL}/runs"
CC = f"{VOL}/yolo"  # images/<split>/<plan>.png, labels/<split>/<plan>.txt

vol = modal.Volume.from_name("stoa-cubicasa", create_if_missing=True)
app = modal.App("stoa-cubicasa-userplans")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics>=8.3", "pillow")
    .add_local_python_source("stoa_qa", ignore=["**/__pycache__/**"])
)


def _predict(weights: list[str], files: list[str], imgsz: int, batch: int = 16) -> dict:
    """{plan: [[cls, score, x0, y0, x1, y1]]} at conf 0.05; two runs are fused like stoa_qa.cubicasa_ensemble."""
    from ultralytics import YOLO

    from stoa_qa.cubicasa_ensemble import fuse

    runs = []
    for w in weights:
        m = YOLO(w)
        out = {}
        for i in range(0, len(files), batch):
            for f, r in zip(files[i:i + batch], m.predict(files[i:i + batch], imgsz=imgsz, conf=0.05, verbose=False)):
                b = r.boxes
                out[Path(f).stem] = [[int(c), round(float(s), 4), *[round(float(v), 1) for v in xyxy]]
                                     for c, s, xyxy in zip(b.cls, b.conf, b.xyxy)]
        runs.append(out)
    return runs[0] if len(runs) == 1 else {k: fuse(runs[0][k], runs[1].get(k, [])) for k in runs[0]}


def _write_user(plans: dict, root: Path, with_labels: bool) -> list[str]:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    files = []
    for name, p in plans.items():
        f = root / "images" / f"{name}{p['ext']}"
        f.write_bytes(p["image"])
        files.append(str(f))
        if with_labels:
            (root / "labels" / f"{name}.txt").write_text(p["label"])
    return sorted(files)


def _weights(runs: list[str]) -> list[str]:
    out = [f"{RUNS}/{r}/weights/best.pt" for r in runs]
    for w in out:
        if not Path(w).exists():
            raise FileNotFoundError(f"{w} is not on the Volume (train it with cubicasa_cloud.py::fit_full first)")
    return out


@app.function(image=image, volumes={VOL: vol}, gpu="L40S", cpu=8, memory=32768, timeout=3600)
def predict_plans(runs: list[str], plans: dict, imgsz: int) -> dict:
    files = _write_user(plans, Path("/tmp/user"), with_labels=False)
    return _predict(_weights(runs), files, imgsz)


@app.function(image=image, volumes={VOL: vol}, gpu="H100", cpu=16, memory=98304, timeout=6 * 3600)
def finetune_remote(name: str, base: str, plans: dict, splits: dict, hp: dict, mix: dict, with_base: bool) -> dict:
    from ultralytics import YOLO

    from stoa_qa import cubicasa_userplans as U

    user = Path("/tmp/user")
    user_files = _write_user(plans, user, with_labels=True)
    cc = {s: sorted(p.stem for p in Path(f"{CC}/labels/{s}").glob("*.txt")) for s in ("train", "val")}
    entries = U.mix_plan(splits, cc, **mix)
    data = U.materialize(entries, user, Path(CC), Path("/tmp/data"))  # copies the sampled CubiCasa plans to local disk
    base_w = _weights([base])[0]
    m = YOLO(base_w)
    m.add_callback("on_fit_epoch_end", lambda trainer: vol.commit())
    m.train(data=str(data), project=RUNS, name=name, exist_ok=True, device=0, workers=12, **U.finetune_args(**hp))
    vol.commit()
    imgsz = hp.get("imgsz", 1280)
    cc_files = {s: sorted(str(p) for p in Path(f"{CC}/images/{s}").glob("*.png")) for s in ("val", "test")}
    out = {"mix": {f"{src}/{s}": sum(1 for e in entries if e[0] == src and e[2] == s)
                   for src in ("user", "cubicasa") for s in ("train", "val", "test")},
           "label_truth": {s: {Path(f).stem: U.label_counts(Path(f"{CC}/labels/{s}/{Path(f).stem}.txt").read_text())
                               for f in cc_files[s]} for s in ("val", "test")},
           "preds": {}}
    for tag, w in ([(name, f"{RUNS}/{name}/weights/best.pt")] + ([(base, base_w)] if with_base else [])):
        out["preds"][tag] = {"user": _predict([w], user_files, imgsz),
                             **{s: _predict([w], cc_files[s], imgsz) for s in ("val", "test")}}
    return out


def _read_plans(root: Path, with_labels: bool) -> dict:
    sys.path.insert(0, str(REPO))
    from stoa_qa import cubicasa_userplans as U

    imgs = U.list_images(root)
    out = {}
    for name, f in imgs.items():
        lbl = root / "labels" / f"{name}.txt"
        if with_labels and not lbl.exists():
            continue
        out[name] = {"image": f.read_bytes(), "ext": f.suffix.lower(), "label": lbl.read_text() if with_labels else ""}
    return out


@app.local_entrypoint()
def label(plans: str, runs: str = "m13-1280", imgsz: int = 1280, overwrite: bool = False, picks: str = ""):
    """Pseudo-label a folder of plans with the run(s) on the Volume (comma-separated runs are fused)."""
    sys.path.insert(0, str(REPO / "modal_jobs"))
    import cubicasa_userplans as L  # the local job: same writers, same file layout

    from stoa_qa import cubicasa_userplans as U

    root = Path(plans)
    run_list = runs.split(",")
    payload = _read_plans(root, with_labels=False)
    if not payload:
        raise SystemExit(f"no plan images in {U.images_dir(root)}")
    preds = predict_plans.remote(run_list, payload, imgsz)
    tag = run_list[0] if len(run_list) == 1 else "ens-" + "-".join(run_list)
    f = L.save_preds(root, tag, preds)
    thr, iou = U.load_picks(Path(picks) if picks else L.PICKS)
    n, boxes = L.write_pseudo_labels(root, preds, thr, iou, overwrite)
    print(f"{len(preds)} plans: raw boxes -> {f}; {n} pseudo-label files ({boxes} boxes) -> {root / 'labels_auto'}")
    print("next: open tools/plan_labeler.html, choose the folder, correct and save each plan (-> labels/)")


@app.local_entrypoint()
def finetune(plans: str, name: str, base: str = "m13-1280", epochs: int = 15, lr0: float = 2e-4, freeze: int = 10,
             imgsz: int = 1280, batch: int = 16, patience: int = 8, repeat: int = 3, replay_ratio: float = 2.0,
             replay_val: int = 100, seed: int = 0, with_base: bool = True):
    root = Path(plans)
    sf = root / "splits.json"
    if not sf.exists():
        raise SystemExit(f"{sf} missing: run  python modal_jobs/cubicasa_userplans.py split {root}  first")
    splits = json.loads(sf.read_text())["plans"]
    payload = _read_plans(root, with_labels=True)
    missing = sorted(set(splits) - set(payload))
    if missing:
        raise SystemExit(f"plans in splits.json without image or corrected label: {missing[:5]}")
    payload = {k: payload[k] for k in splits}
    hp = dict(epochs=epochs, lr0=lr0, freeze=freeze, imgsz=imgsz, batch=batch, patience=patience)
    mix = dict(repeat=repeat, replay_ratio=replay_ratio, replay_val=replay_val, seed=seed)
    print(f"uploading {len(payload)} corrected plans; fine-tuning runs/{base} -> runs/{name}: {hp} {mix}")
    out = finetune_remote.remote(name, base, payload, splits, hp, mix, with_base)
    print("dataset:", out["mix"])
    for tag, p in out["preds"].items():
        (root / "predictions").mkdir(exist_ok=True)
        (root / "predictions" / f"{tag}.json").write_text(json.dumps(p["user"]))
        dest = REPO / "eval_out/yolo" / tag
        dest.mkdir(parents=True, exist_ok=True)
        for s in ("val", "test"):
            f = dest / f"{s}.json"
            if tag == name or not f.exists():  # never overwrite an existing base run's boxes
                f.write_text(json.dumps(p[s]))
        print(f"{tag}: {len(p['user'])} user plans, {len(p['val'])} val / {len(p['test'])} test CubiCasa plans -> {dest}")
    (REPO / "eval_out/yolo" / name / "label_truth.json").write_text(json.dumps(out["label_truth"]))
    print(f"next: python modal_jobs/cubicasa_userplans.py evaluate {root} --before {base} --after {name}")
