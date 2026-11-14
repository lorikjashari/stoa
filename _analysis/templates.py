import json, collections, os, re, sys

ROOT = r"c:\Users\diar\Desktop\stoa\junction-dataset\junction-dataset"
rows = [json.loads(l) for l in open(os.path.join(ROOT, "questions.jsonl"), encoding="utf-8")]
only = sys.argv[1] if len(sys.argv) > 1 else None
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 80

def norm(q):
    q = re.sub(r'"[^"]*"', '"<X>"', q)
    q = re.sub(r"'[^']*'", "'<X>'", q)
    q = re.sub(r"«[^»]*»", "«<X>»", q)
    q = re.sub(r"„[^“]*“", "„<X>“", q)
    q = re.sub(r"\b\d+(\.\d+)?\b", "<N>", q)
    return q

by_src = collections.defaultdict(list)
for r in rows:
    by_src[r["path"].split("/")[1]].append(r)

for s, rs in sorted(by_src.items()):
    if only and s != only:
        continue
    t = collections.Counter((r["type"], r["unit"], norm(r["question"])) for r in rs)
    print(f"\n######## {s}: {len(t)} templates, {len({r['question'] for r in rs})} unique questions")
    for (ty, un, q), n in t.most_common(limit):
        print(f"{n:6} [{ty}/{un}] {q}")
