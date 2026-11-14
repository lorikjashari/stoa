"""How do Stoa's cubicasa5k answers follow from CubiCasa's model.svg annotations? TRAIN split only.

Reads eval_out/cubicasa_objects.json.gz (modal_jobs/cubicasa_local.py extract). Prints, per room question,
the Space classes / name labels of the rooms in plans where the annotation count is ambiguous, and how often
candidate rules reproduce Stoa's answer. The rules found here become detector classes; val and test are not read.
"""
import collections
import gzip
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from stoa_qa.cubicasa_gemini import load_truth  # noqa: E402
from stoa_qa.cubicasa_rules import QUESTION_CLASSES, plan_boxes  # noqa: E402

plans = [p for p in json.loads(gzip.decompress((REPO / "eval_out/cubicasa_objects.json.gz").read_bytes()))
         if "objs" in p and p["split"] == "train"]
truth = load_truth()
print(len(plans), "train plans")

space = collections.Counter()
for p in plans:
    for o in p["objs"]:
        if o["c"].startswith("Space "):
            space[o["c"]] += 1
print("Space classes:", space.most_common(40))

# fixtures / openings / floors with the current rules
for q, classes in QUESTION_CLASSES.items():
    ok = n = 0
    for p in plans:
        gt = truth.get(f"data/cubicasa5k/{p['name']}.png", {}).get(q)
        if gt is None:
            continue
        c = collections.Counter(lab for lab, _ in plan_boxes(p))
        n += 1
        ok += sum(c[k] for k in classes) == gt
    print(f"  rule {q:9}: {ok}/{n} = {ok / max(n, 1):.1%}")

ROOMQ = ["saunas", "kitchens", "bedrooms", "wc_rooms", "bathrooms", "balconies"]
# for each room question: which (class, label) pairs co-occur with the answer? fit "count of class in S"
pairs = collections.defaultdict(collections.Counter)
for p in plans:
    gt = truth.get(f"data/cubicasa5k/{p['name']}.png", {})
    rooms = [(o["c"].split(" ", 1)[1], re.sub(r"\s+", " ", (o.get("t") or "").upper()).strip()) for o in p["objs"] if o["c"].startswith("Space ")]
    for q in ROOMQ:
        if q in gt:
            for cls, _ in rooms:
                pairs[q][cls] += 0  # register
for q in ROOMQ:
    # greedy: per Space class, does counting it alone explain the answer?
    cls_all = {c.split(" ", 1)[1] for c in space}
    best = []
    for c in cls_all:
        ok = n = 0
        for p in plans:
            gt = truth.get(f"data/cubicasa5k/{p['name']}.png", {}).get(q)
            if gt is None:
                continue
            n += 1
            ok += sum(o["c"] == f"Space {c}" for o in p["objs"]) == gt
        best.append((ok / max(n, 1), c, n))
    best.sort(reverse=True)
    print(q, [(f"{a:.1%}", c) for a, c, _ in best[:4]])
