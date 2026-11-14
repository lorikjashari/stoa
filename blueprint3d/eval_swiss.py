"""Score the DXF -> scene extractor against Stoa's ground truth for swiss_dwellings (English questions).

    py -I eval_swiss.py [--limit N]

Uses the same scene the 3D viewer renders, so this number is also the accuracy of what the 3D model shows.
Counts must be exact; areas within 2% (the challenge's scoring rule).
"""
import argparse, collections, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dxf_to_scene import DEFAULTS, build_scene

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "junction-dataset", "junction-dataset")


def answer(scene, q):
    R = scene["rooms"]
    indoor = [r for r in R if r["number"] and not r["outdoor"]]
    total = lambda rs: round(sum(r["area_m2"] for r in rs), 2)
    if q.startswith("How many doors"):
        return "doors", len(scene["doors"])
    if q.startswith("How many windows"):
        return "windows", len(scene["windows"])
    if q.startswith("How many apartments"):
        return "apartments", scene["counts"]["units"]
    if m := re.match(r"How many rooms labelled '(.+)' are there", q):
        return "rooms by label", sum(1 for r in R if r["label"] == m[1])
    if m := re.match(r"What is the net floor area of room (\S+) \('(.+)'\)", q):
        hit = [r for r in R if r["number"] == m[1]]
        return "room area", hit[0]["area_m2"] if hit else None
    if m := re.match(r"What is the net floor area of apartment ([A-Z]) ", q):
        u = next((u for u in scene["units"] if u["label"] and u["label"].rstrip().endswith(m[1])), None)
        return "apartment area", total([r for r in indoor if u and r["unit"] == u["id"]]) if u else None
    if m := re.match(r"What is the total net area in m² of all rooms labelled '(.+)'", q):
        return "label total area", total([r for r in R if r["label"] == m[1]])
    if q.startswith("What is the total net floor area in m² of all numbered indoor rooms"):
        return "floor indoor area", total(indoor)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    by_file = collections.defaultdict(list)
    for line in open(os.path.join(ROOT, "questions.jsonl"), encoding="utf-8"):
        row = json.loads(line)
        if "/swiss_dwellings/" in row["path"]:
            by_file[row["path"]].append(row)
    files = sorted(by_file)[: a.limit or None]
    stats = collections.defaultdict(lambda: [0, 0])
    errors = collections.Counter()
    for path in files:
        dxf = os.path.join(ROOT, path[:-4] + ".dxf")
        try:
            scene = build_scene(dxf, DEFAULTS)
        except Exception as e:  # report, don't hide
            errors[type(e).__name__] += 1
            continue
        for row in by_file[path]:
            kind, pred = answer(scene, row["question"])
            if kind is None:
                continue
            ans, s = row["answer"], stats[kind]
            s[1] += 1
            if pred is None:
                continue
            ok = pred == ans if row["unit"] == "count" else (ans != 0 and abs(pred - ans) / abs(ans) <= 0.02)
            s[0] += ok
    print(f"{len(files)} floor plans\n")
    print(f"{'question type':<20}{'correct':>9}{'total':>8}{'acc':>8}")
    tc = tt = 0
    for k, (c, t) in sorted(stats.items(), key=lambda kv: -kv[1][1]):
        print(f"{k:<20}{c:>9}{t:>8}{c / t:>8.1%}")
        tc += c; tt += t
    print(f"{'ALL':<20}{tc:>9}{tt:>8}{tc / tt:>8.1%}")
    if errors:
        print("parse errors:", dict(errors))


if __name__ == "__main__":
    main()
