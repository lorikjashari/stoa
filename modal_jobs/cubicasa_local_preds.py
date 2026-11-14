"""Cache the pretrained CubiCasa5K network's prediction for every CubiCasa plan, on the local GPU.

Writes third_party/cubicasa_preds/<stem>.json in exactly the shape blueprint3d/eval_cubicasa_model.detect()
writes (it *is* detect(): same model, weights, post-processing and thresholds). The only difference is
that the network's forward pass runs on CUDA; the output goes back to CPU before the repo's post-processing.
TF32 is turned off so the GPU forward pass stays fp32 like the CPU one.

Resumable: plans whose cache file exists are skipped; each file is written to a staging folder first and
moved into place with os.replace, so a crash never leaves a half-written cache file.

    .venv/Scripts/python.exe -W ignore modal_jobs/cubicasa_local_preds.py --limit 5        # measure
    .venv/Scripts/python.exe -W ignore modal_jobs/cubicasa_local_preds.py --workers 4      # everything
    .venv/Scripts/python.exe -W ignore modal_jobs/cubicasa_local_preds.py --only colorful-10052 --check
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
B3D = REPO / "blueprint3d"
PLANS = REPO / "junction-dataset/junction-dataset/data/cubicasa5k"
CACHE = REPO / "third_party/cubicasa_preds"
STAGE = REPO / "eval_out/cubicasa_local_preds/stage"
LOG_DIR = REPO / "eval_out/cubicasa_local_preds"


class _OnGPU:
    """Stands in for the model inside cubicasa_model.predict(): forward on CUDA, result back on CPU."""

    def __init__(self, model, device):
        self.model, self.device = model.to(device), device

    def __call__(self, x):
        import torch
        with torch.no_grad():
            return self.model(x.to(self.device, non_blocking=True)).float().cpu()


def _init_worker(device: str):
    import torch
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(B3D))
    import cubicasa_model as M
    import eval_cubicasa_model as E
    if device != "cpu":
        M._model = _OnGPU(M.load_model(), device)
    E.CACHE = str(STAGE / str(os.getpid()))  # detect() writes here; we then move the file into the real cache
    os.makedirs(E.CACHE, exist_ok=True)


def _run_one(stem: str) -> dict:
    import eval_cubicasa_model as E
    rel = f"data/cubicasa5k/{stem}.png"
    staged = Path(E.CACHE) / f"{stem}.json"
    final = CACHE / f"{stem}.json"
    if final.exists():
        return {"stem": stem, "skipped": True}
    staged.unlink(missing_ok=True)
    t = time.time()
    try:
        res = E.detect(rel)
        json.loads(staged.read_text(encoding="utf-8"))  # written completely?
        os.replace(staged, final)
        return {"stem": stem, "seconds": round(time.time() - t, 1), "counts": res["counts"],
                "walls": len(res["walls"]), "rooms": len(res["rooms"])}
    except Exception as e:
        staged.unlink(missing_ok=True)
        if "CUDA" in repr(e) and _to_cpu():
            # A CUDA fault poisons this process's GPU context for good; finish this worker's plans on CPU
            # (same weights from disk; CPU and GPU outputs were identical on the plans checked).
            r = _run_one(stem)
            r["cuda_fallback"] = repr(e)[:120]
            return r
        return {"stem": stem, "error": repr(e)[:300], "trace": traceback.format_exc()[-1500:]}


def _to_cpu() -> bool:
    """Swap this worker's model for a fresh CPU copy; False if it is already on CPU."""
    import cubicasa_model as M
    if not isinstance(M._model, _OnGPU):
        return False
    M._model = None
    M.load_model()  # loads the checkpoint with map_location="cpu", never touches CUDA
    print(f"  worker {os.getpid()}: CUDA error, switched to CPU", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3, help="processes sharing the GPU (post-processing is CPU-bound)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*", help="plan stems to (re)run into the staging folder, for --check")
    ap.add_argument("--check", action="store_true",
                    help="with --only: recompute already-cached plans on GPU and compare with the cache, write nothing")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if a.check:
        _init_worker(a.device)
        import eval_cubicasa_model as E
        for stem in a.only or []:
            old = json.loads((CACHE / f"{stem}.json").read_text(encoding="utf-8"))
            (Path(E.CACHE) / f"{stem}.json").unlink(missing_ok=True)
            t = time.time()
            new = E.detect(f"data/cubicasa5k/{stem}.png")
            (Path(E.CACHE) / f"{stem}.json").unlink(missing_ok=True)
            same = {k: old[k] == new[k] for k in ("counts", "icons", "walls", "rooms")}
            print(f"{stem}: {time.time() - t:.1f}s  identical={same}  old={old['counts']}  new={new['counts']}", flush=True)
        return

    stems = sorted(p.stem for p in PLANS.glob("*.png"))
    if a.only:
        stems = [s for s in stems if s in set(a.only)]
    todo = [s for s in stems if not (CACHE / f"{s}.json").exists()]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(stems)} plans, {len(stems) - len(todo) if not a.limit else '-'} cached, {len(todo)} to run "
          f"on {a.device} with {a.workers} workers", flush=True)
    if not todo:
        return

    log = open(LOG_DIR / "results.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    done = errors = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init_worker, initargs=(a.device,), maxtasksperchild=200) as pool:
        for r in pool.imap_unordered(_run_one, todo):
            log.write(json.dumps(r) + "\n"); log.flush()
            if r.get("skipped"):
                continue
            if "error" in r:
                errors += 1
                print(f"  ERROR {r['stem']}: {r['error']}", flush=True)
            else:
                done += 1
            n = done + errors
            if n % 25 == 0 or n == len(todo) or n <= 3:
                el = time.time() - t0
                eta = el / n * (len(todo) - n)
                print(f"  {n}/{len(todo)}  ok={done} err={errors}  {el / 60:.1f} min  {el / n:.2f} s/plan  "
                      f"eta {eta / 60:.0f} min  last={r['stem']} {r.get('seconds', '')}s", flush=True)
    el = time.time() - t0
    total = sum(1 for s in stems if (CACHE / f"{s}.json").exists())
    print(f"finished: {done} new, {errors} errors in {el / 60:.1f} min; {total}/{len(stems)} plans cached", flush=True)


if __name__ == "__main__":
    main()
