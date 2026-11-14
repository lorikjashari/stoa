import json, os, re, collections
import swiss_dxf as S

ROOT = r"c:\Users\diar\Desktop\stoa\junction-dataset\junction-dataset"
rows = [json.loads(l) for l in open(os.path.join(ROOT, "questions.jsonl"), encoding="utf-8")]
KEYS = {"toilets": "How many toilets", "basins": "How many washbasins", "tubs": "How many bathtubs", "showers": "How many showers"}
ans = collections.defaultdict(dict)
paths = set()
for r in rows:
    if "/swiss_dwellings/" in r["path"]:
        paths.add(r["path"])
        for k, pre in KEYS.items():
            if r["question"].startswith(pre):
                ans[r["path"]][k] = r["answer"]


def feats(ents):
    f = collections.Counter()
    for e in ents:
        if e["layer"] != "FIXTURE":
            continue
        if e["type"] == "CIRCLE":
            f[f"circle_r{e['r']:.2f}"] += 1
        elif e["type"] == "LWPOLYLINE":
            n = len(e["pts"])
            closed = e.get("flags", 0) & 1
            if n == 2:
                f["line"] += 1
            elif n in (4,) and closed:
                f["rect"] += 1
            else:
                f[f"poly{n}{'c' if closed else ''}"] += 1
    return f


data = []
for p in sorted(paths):
    ents = S.read_entities(os.path.join(ROOT, p[:-4] + ".dxf"))
    data.append((p, feats(ents), ans[p]))

allf = collections.Counter()
for _, f, _ in data:
    allf.update(f.keys())
print("feature prevalence:", allf.most_common(40))

# for each target, find single features that equal the answer most often
for k in KEYS:
    score = collections.Counter()
    n = 0
    for p, f, a in data:
        if k not in a:
            continue
        n += 1
        for feat in allf:
            if f.get(feat, 0) == a[k]:
                score[feat] += 1
    print(k, n, [(ft, round(s / n, 3)) for ft, s in score.most_common(6)])
