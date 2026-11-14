import json, collections, os, re, statistics as st

ROOT = r"c:\Users\diar\Desktop\stoa\junction-dataset\junction-dataset"
rows = [json.loads(l) for l in open(os.path.join(ROOT, "questions.jsonl"), encoding="utf-8")]

def norm(q):
    q = re.sub(r"'[^']*(?:'[^']*)?'(?= are| in| on|\))", "'<X>'", q)
    q = re.sub(r"«[^»]*»", "«<X>»", q)
    q = re.sub(r"„[^“”]*[“”]", "„<X>“", q)
    q = re.sub(r"\b(room|Raum|pièce|locale) (U\d\.)?\d+", r"\1 <N>", q)
    q = re.sub(r"apartment [A-Z]\b", "apartment <L>", q)
    return q

def summarize(vals):
    vals = sorted(vals)
    n = len(vals)
    return f"n={n} min={vals[0]} p50={vals[n//2]} mean={st.mean(vals):.2f} max={vals[-1]}"

for s in ["cubicasa5k", "swiss_dwellings", "ifc_models"]:
    print(f"\n######## {s}")
    g = collections.defaultdict(list)
    for r in rows:
        if r["path"].split("/")[1] != s or r["type"] == "text":
            continue
        q = norm(r["question"])
        if not q.isascii() and s != "cubicasa5k":
            pass
        g[q].append(r["answer"])
    for q, v in sorted(g.items(), key=lambda kv: -len(kv[1]))[:40]:
        extra = ""
        if all(isinstance(x, int) for x in v):
            extra = " dist=" + str(sorted(collections.Counter(v).items())[:15])
        print(f"{summarize(v)} | {q[:140]}{extra}")

print("\n== decimal places of area answers per source")
for s in ["swiss_dwellings", "ifc_models"]:
    d = collections.Counter(len(str(r["answer"]).split(".")[1]) if "." in str(r["answer"]) else 0
                            for r in rows if r["path"].split("/")[1] == s and r["type"] in ("area", "volume", "length"))
    print(s, d)

print("\n== swiss labels in 'rooms labelled' questions (EN)")
lab = collections.Counter()
for r in rows:
    m = re.match(r"How many rooms labelled '(.+)' are there on this floor\?", r["question"])
    if m:
        lab[m.group(1)] += 1
print(len(lab), lab.most_common(80))
