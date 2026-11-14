"""Score blueprint3d/archive_plan's floor-plan check against the hand labels.

    .venv/Scripts/python.exe -W ignore _analysis/archive_eval.py [archive_sample ...] [--cached]

Labels (_analysis/archive_labels*.json): plan / multi (a plan beside other drawings) / site / other.
Accepting a 'plan' or 'multi' sheet is right only if the chosen region is a floor plan: check by eye on the saved
crops eval_out/archive_eval/<seed>_<source>_<k>_<label>.jpg. Per-view features are cached in
eval_out/archive_eval/<sample>.json, so with --cached a threshold change re-scores without the network.
"""
import json, os, sys, time
import cv2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "blueprint3d"))
import archive_plan as A
ROOT = os.path.join(HERE, "..", "junction-dataset", "junction-dataset", "data")
OUT = os.path.join(HERE, "..", "eval_out", "archive_eval")
os.makedirs(OUT, exist_ok=True)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
cached = "--cached" in sys.argv
samples = args or ["archive_sample", "archive_sample_11"]

tot = {}
for sn in samples:
    S = json.load(open(os.path.join(HERE, sn + ".json")))
    L = json.load(open(os.path.join(HERE, sn.replace("sample", "labels") + ".json")))
    cache_p = os.path.join(OUT, sn + ".json")
    C = json.load(open(cache_p)) if cached and os.path.exists(cache_p) else {}
    for src, items in S.items():
        for k, it in enumerate(items):
            key = f"{src}/{it['file']}"
            p = os.path.join(ROOT, src, it["file"])
            if key not in C:
                t = time.time()
                r = A.analyse_sheet(p, keep_pred=False)
                C[key] = {"views": [{kk: v[kk] for kk in ("box", "parent", "children", "features", "labels")} for v in r["views"]],
                          "sec": round(time.time() - t, 1)}
            views = C[key]["views"]
            best, nplans = A.judge(views)
            lab = L[src][k]
            row = tot.setdefault(src, {}).setdefault(lab, [0, 0])
            row[0] += 1
            row[1] += best is not None
            vs = " ".join(f"{i}:{v['verdict'][0]}" for i, v in enumerate(views))
            print(f"{sn[-2:]} {src:9s} {k:2d} {lab:5s} {'ACCEPT' if best is not None else 'reject'} best={best} plans={nplans} [{vs}] {C[key]['sec']}s")
            if best is not None:
                rgb = A.load_rgb(p)
                x0, y0, x1, y1 = views[best]["box"]
                crop = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
                s = 500 / max(crop.shape[:2])
                cv2.imwrite(os.path.join(OUT, f"{sn[-2:]}_{src}_{k:02d}_{lab}.jpg"),
                            cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    json.dump(C, open(cache_p, "w"))
print("\nper source: label -> accepted / total")
for src, row in tot.items():
    print(f"  {src:9s} " + "  ".join(f"{lab}: {a}/{n}" for lab, (n, a) in sorted(row.items())))
