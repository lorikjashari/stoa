"""Score the raster (cubicasa5k) pipeline on door and window counts against Stoa's ground truth.

    .venv/Scripts/python blueprint3d/eval_raster.py [--per-style 50] [--seed 0]

Counts must be exact (the challenge rule); we also report within ±1 to show how close the geometry is.
"""
import argparse, collections, json, os, random, sys, time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "junction-dataset", "junction-dataset")


def run(path):
    import raster_to_scene as R
    try:
        A = R.analyse(os.path.join(ROOT, path))
        c = collections.Counter(g["kind"] for g in A["gaps"])
        return path, c["door"], c["window"], len(A["rooms"])
    except Exception as e:
        return path, None, None, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-style", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    truth = collections.defaultdict(dict)
    for line in open(os.path.join(ROOT, "questions.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if "/cubicasa5k/" not in r["path"]:
            continue
        q = r["question"]
        if q.startswith("How many doors"):
            truth[r["path"]]["doors"] = r["answer"]
        elif q.startswith("How many windows"):
            truth[r["path"]]["windows"] = r["answer"]
    by_style = collections.defaultdict(list)
    for p in sorted(truth):
        by_style[os.path.basename(p).rsplit("-", 1)[0]].append(p)
    rnd = random.Random(a.seed)
    sample = [p for s in sorted(by_style) for p in rnd.sample(by_style[s], min(a.per_style, len(by_style[s])))]
    t0 = time.time()
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(run, sample))
    stats = collections.defaultdict(lambda: [0, 0, 0])  # exact, within1, total
    errors = 0
    for path, d, w, _ in results:
        if d is None:
            errors += 1
            continue
        style = os.path.basename(path).rsplit("-", 1)[0]
        for k, pred in (("doors", d), ("windows", w)):
            if k in truth[path]:
                s = stats[(style, k)]
                s[2] += 1; s[0] += pred == truth[path][k]; s[1] += abs(pred - truth[path][k]) <= 1
    print(f"{len(sample)} plans in {time.time() - t0:.0f}s ({errors} failed)\n")
    print(f"{'style':<28}{'question':<10}{'exact':>8}{'±1':>8}{'n':>6}")
    for (style, k), (e, w1, n) in sorted(stats.items()):
        print(f"{style:<28}{k:<10}{e / n:>8.0%}{w1 / n:>8.0%}{n:>6}")


if __name__ == "__main__":
    main()
