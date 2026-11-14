"""Score the IFC scene against Stoa's ground truth for ifc_models (English questions).

    python blueprint3d/eval_ifc.py [--verbose]

The scene is built by the same IFC reader that answers questions (stoa_qa.ifc).
Counts must be exact; areas, lengths and volumes within 2%.
"""
import argparse, collections, json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ifc_to_scene import build_scene

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "junction-dataset", "junction-dataset")
GROUPS = {"doors": "doors", "windows": "windows", "walls": "walls", "stairs": "stairs", "columns": "columns",
          "beams": "beams", "slabs": "slabs", "sanitary fixtures": "sanitary", "light fixtures": "lights"}


def model_path(path):
    base = re.sub(r"_(plan|elevation|section)-.*$", "", path.rsplit(".", 1)[0])
    return os.path.join(ROOT, base + ".ifc")


def answer(scene, q):
    rooms, els = scene["rooms"], scene["elements"]
    st = {s["name"]: s["id"] for s in scene["storeys"]}
    on = lambda xs, name: [x for x in xs if x["storey"] == st.get(name)]
    total = lambda xs, k: round(sum(x[k] for x in xs if x.get(k) is not None), 4) if any(x.get(k) is not None for x in xs) else None
    if m := re.match(r"How many (\w[\w ]*?) are shown on the floor plan of '(.+)'\?", q):
        what, s = m[1], m[2]
        if what.startswith("rooms"):
            return "count/storey rooms", len(on(rooms, s))
        g = GROUPS.get(what)
        return f"count/storey {g}", len([e for e in on(els, s) if e["group"] == g]) if g else None
    if m := re.match(r"How many (\w[\w ]*?) are there in the model '(.+)'\?", q):
        what = m[1]
        if what.startswith("rooms"):
            return "count/model rooms", len(rooms)
        g = GROUPS.get(what)
        return f"count/model {g}", len([e for e in els if e["group"] == g]) if g else None
    if m := re.match(r"What is the net floor area of the room '(.+)' on '(.+)', in m²\?", q):
        hit = [r for r in on(rooms, m[2]) if r["label"] == m[1]]
        return "area/room", hit[0]["net_floor_area_m2"] if len(hit) == 1 else None
    if m := re.match(r"What is the total net floor area of all rooms \(spaces\) on '(.+)', in m²\?", q):
        return "area/storey rooms", total(on(rooms, m[1]), "net_floor_area_m2")
    if m := re.match(r"What is the total net volume of all slabs on '(.+)', in m³\?", q):
        return "volume/storey slabs", total([e for e in on(els, m[1]) if e["group"] == "slabs"], "net_volume_m3")
    if m := re.match(r"What is the total net volume of all walls on '(.+)', in m³\?", q):
        return "volume/storey walls", total([e for e in on(els, m[1]) if e["group"] == "walls"], "net_volume_m3")
    if m := re.match(r"What is the total length of all walls on '(.+)', in metres\?", q):
        return "length/storey walls", total([e for e in on(els, m[1]) if e["group"] == "walls"], "length_m")
    if m := re.match(r"What is the total net side area of all walls on '(.+)', in m²\?", q):
        return "area/storey walls", total([e for e in on(els, m[1]) if e["group"] == "walls"], "net_side_area_m2")
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "questions.jsonl"), encoding="utf-8") if "/ifc_models/" in l]
    by_model = collections.defaultdict(list)
    for r in rows:
        by_model[model_path(r["path"])].append(r)
    stats = collections.defaultdict(lambda: [0, 0, 0])  # correct, answered, total
    per_model = {}
    for path, qs in sorted(by_model.items()):
        t0 = time.time()
        scene = build_scene(path)
        good = tot = 0
        for r in qs:
            kind, pred = answer(scene, r["question"])
            if kind is None:
                continue
            s = stats[kind]; s[2] += 1; tot += 1
            if pred is None:
                continue
            s[1] += 1
            ans = r["answer"]
            ok = pred == ans if r["unit"] == "count" else (ans != 0 and abs(pred - ans) / abs(ans) <= 0.02) or (ans == 0 and pred == 0)
            s[0] += ok; good += ok
            if a.verbose and not ok:
                print(f"  x {os.path.basename(path)[:40]} | {r['question'][:90]} | gold {ans} | ours {pred}")
        per_model[os.path.basename(path)] = (good, tot, round(time.time() - t0, 1))
    print(f"{len(by_model)} IFC models\n")
    print(f"{'question type':<24}{'correct':>8}{'answered':>10}{'total':>7}{'acc':>8}")
    C = T = 0
    for k, (c, n, t) in sorted(stats.items(), key=lambda kv: -kv[1][2]):
        print(f"{k:<24}{c:>8}{n:>10}{t:>7}{c / t:>8.1%}")
        C += c; T += t
    print(f"{'ALL':<24}{C:>8}{'':>10}{T:>7}{C / T:>8.1%}\n")
    for m, (g, t, s) in sorted(per_model.items(), key=lambda kv: -kv[1][1]):
        print(f"  {m[:55]:<56}{g:>4}/{t:<4} {s}s")


if __name__ == "__main__":
    main()
