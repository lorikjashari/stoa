"""Build scenes for every sheet the plan check accepts in the labelled samples; report wall-trace coverage.

    .venv/Scripts/python.exe -W ignore _analysis/archive_build_all.py <out_dir>   (MIN_WALL_COVERAGE forced to 0)
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "blueprint3d"))
import archive_plan as A, raster_to_scene as R
R.MIN_WALL_COVERAGE = 0.0
ROOT = os.path.join(HERE, "..", "junction-dataset", "junction-dataset", "data")
out = sys.argv[1]
for sn in ["archive_sample", "archive_sample_11"]:
    S = json.load(open(os.path.join(HERE, sn + ".json")))
    L = json.load(open(os.path.join(HERE, sn.replace("sample", "labels") + ".json")))
    C = json.load(open(os.path.join(HERE, "..", "eval_out", "archive_eval", sn + ".json")))
    for src, items in S.items():
        for k, it in enumerate(items):
            if A.judge(C[f"{src}/{it['file']}"]["views"])[0] is None:
                continue
            t = time.time()
            try:
                sc = R.build(os.path.join(ROOT, src, it["file"]), out)
                s, c = sc["source"], sc["counts"]
                b = sc["bbox"]
                print(f"{sn[-2:]} {src:9s} {k:2d} {L[src][k]:5s} cov={s['wall_trace_coverage']:.2f} rooms={c['rooms']:2d} walls={c['walls']:3d} "
                      f"doors={c['doors']:2d} win={c['windows']:2d} {b[2]-b[0]:.0f}x{b[3]-b[1]:.0f} m  scale: {s['scale_from'][:40]}  {time.time()-t:.0f}s", flush=True)
            except R.NotAPlan as e:
                print(f"{sn[-2:]} {src:9s} {k:2d} {L[src][k]:5s} NotAPlan({e.kind}): {e.reason}", flush=True)
