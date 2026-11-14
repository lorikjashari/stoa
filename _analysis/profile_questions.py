import json, collections, os, re

ROOT = r"c:\Users\diar\Desktop\stoa\junction-dataset\junction-dataset"
rows = [json.loads(l) for l in open(os.path.join(ROOT, "questions.jsonl"), encoding="utf-8")]
print("rows", len(rows))
print("keys", collections.Counter(tuple(sorted(r)) for r in rows))

src = lambda r: r["path"].split("/")[1]
print("\n== per source: rows, unique files")
by_src = collections.defaultdict(list)
for r in rows:
    by_src[src(r)].append(r)
for s, rs in sorted(by_src.items()):
    print(f"{s:16} rows={len(rs):6} files={len({r['path'] for r in rs}):5}")

print("\n== type x unit")
print(collections.Counter((r["type"], r["unit"]) for r in rows).most_common())

print("\n== type per source")
for s, rs in sorted(by_src.items()):
    print(s, collections.Counter(r["type"] for r in rs).most_common())

print("\n== extensions with questions")
print(collections.Counter(os.path.splitext(r["path"])[1] for r in rows).most_common())

print("\n== answer python types")
print(collections.Counter((r["type"], type(r["answer"]).__name__) for r in rows).most_common())

missing = {r["path"] for r in rows if not os.path.exists(os.path.join(ROOT, r["path"]))}
print("\nmissing files referenced:", len(missing), list(missing)[:5])

total_files = sum(len(fs) for _, _, fs in os.walk(os.path.join(ROOT, "data")))
print("total files in data:", total_files, "files with questions:", len({r['path'] for r in rows}))

print("\n== questions per file distribution")
c = collections.Counter(r["path"] for r in rows)
print(collections.Counter(c.values()).most_common(20))
