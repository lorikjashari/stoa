"""Score the pretrained CubiCasa5K model on Stoa's cubicasa5k questions, honestly split by train/test.

    .venv/Scripts/python blueprint3d/eval_cubicasa_model.py [--split test] [--limit N] [--workers 4]

The model was trained on CubiCasa's train split; Stoa's answers likely come from the same annotations.
Only the TEST split (plans the model never saw) is a fair number. Predictions are cached per plan in
third_party/cubicasa_preds/<name>.json so the scene builder and re-runs reuse them.
"""
import argparse, collections, json, os, re, sys, time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "junction-dataset", "junction-dataset")
SPLITS = os.path.join(HERE, "..", "third_party", "cubicasa_split")
CACHE = os.path.join(HERE, "..", "third_party", "cubicasa_preds")


def split_members(split):
    rows = open(os.path.join(SPLITS, f"cubicasa5k__{split}.txt"), encoding="utf-8").read().split()
    return {f"data/cubicasa5k/{r.strip('/').replace('/', '-')}.png" for r in rows}


def detect(path):
    """Model prediction -> instance counts + polygons (cached)."""
    name = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(CACHE, name + ".json")
    if os.path.exists(out):
        return json.load(open(out, encoding="utf-8"))
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 4))
    sys.path.insert(0, HERE)
    import cubicasa_model as M
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union
    t = time.time()
    hm, rm, ic, scale = M.predict(os.path.join(ROOT, path))
    polygons, types, room_polygons, room_types = M.vectorize(hm, rm, ic)
    icons, walls = [], []
    for poly, ty in zip(polygons, types):
        pts = [[round(float(x) / scale, 1), round(float(y) / scale, 1)] for x, y in poly]
        if ty["type"] == "icon":
            icons.append({"class": M.ICON_CLASSES[ty["class"]], "poly": pts})
        else:
            walls.append({"class": int(ty["class"]), "poly": pts})
    rooms = []
    for geom, ty in zip(room_polygons, room_types):
        parts = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
        for g in parts:
            if g.area * 1.0 / scale ** 2 < 400:  # ignore slivers (< 20x20 px at original resolution)
                continue
            rooms.append({"class": M.ROOM_CLASSES[ty["class"]],
                          "poly": [[round(x / scale, 1), round(y / scale, 1)] for x, y in g.exterior.coords[:-1]],
                          "holes": [[[round(x / scale, 1), round(y / scale, 1)] for x, y in r.coords[:-1]] for r in g.interiors]})
    # a sauna has one or more benches: count clusters of benches, not benches
    benches = [Polygon(i["poly"]).buffer(30) for i in icons if i["class"] == "Sauna Bench" and len(i["poly"]) >= 3]
    u = unary_union(benches) if benches else None
    saunas = 0 if u is None else (len(u.geoms) if hasattr(u, "geoms") else 1)
    c = collections.Counter(i["class"] for i in icons)
    rc = collections.Counter(r["class"] for r in rooms)
    res = {"path": path, "seconds": round(time.time() - t, 1), "scale": scale,
           "counts": {"doors": c["Door"], "windows": c["Window"], "toilets": c["Toilet"], "sinks": c["Sink"],
                      "baths": c["Bathtub"], "saunas": saunas, "kitchens": rc["Kitchen"], "bedrooms": rc["Bed Room"],
                      "bathrooms": rc["Bath"], "balconies": rc["Outdoor"]},
           "icons": icons, "walls": walls, "rooms": rooms}
    os.makedirs(CACHE, exist_ok=True)
    json.dump(res, open(out, "w", encoding="utf-8"), separators=(",", ":"))
    return res


QUESTIONS = [  # (key, English question prefix)
    ("doors", "How many doors"), ("windows", "How many windows"), ("toilets", "How many toilet bowls"),
    ("sinks", "How many sinks and wash basins"), ("baths", "How many showers and bathtubs"),
    ("saunas", "How many saunas"), ("kitchens", "How many kitchens"), ("bedrooms", "How many bedrooms"),
    ("bathrooms", "How many bathrooms or washrooms"), ("balconies", "How many balconies"),
]


def safe_detect(p):
    try:
        return detect(p)
    except Exception as e:
        return {"path": p, "error": repr(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    truth = collections.defaultdict(dict)
    for line in open(os.path.join(ROOT, "questions.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if "/cubicasa5k/" in r["path"]:
            for k, prefix in QUESTIONS:
                if r["question"].startswith(prefix):
                    truth[r["path"]][k] = r["answer"]
    members = sorted(p for p in split_members(a.split) if p in truth)
    if a.limit:
        members = members[: a.limit]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(safe_detect, members))
    ok = [r for r in results if "error" in r]
    stats = collections.defaultdict(lambda: [0, 0, 0])
    for r in results:
        if "error" in r:
            continue
        for k, _ in QUESTIONS:
            if k in truth[r["path"]]:
                s, pred, gold = stats[k], r["counts"][k], truth[r["path"]][k]
                s[2] += 1; s[0] += pred == gold; s[1] += abs(pred - gold) <= 1
    print(f"split={a.split}: {len(members)} plans in {time.time() - t0:.0f}s, {len(ok)} errors")
    for r in ok[:3]:
        print("  error:", r["path"], r["error"])
    print(f"\n{'question':<12}{'exact':>8}{'±1':>8}{'n':>6}")
    E = N = 0
    for k, _ in QUESTIONS:
        e, w1, n = stats[k]
        if n:
            print(f"{k:<12}{e / n:>8.1%}{w1 / n:>8.1%}{n:>6}")
            E += e; N += n
    print(f"{'ALL':<12}{E / N:>8.1%}{'':>8}{N:>6}")


if __name__ == "__main__":
    main()
