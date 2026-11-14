import json, collections, os, re, random

ROOT = r"c:\Users\diar\Desktop\stoa\junction-dataset\junction-dataset"
DATA = os.path.join(ROOT, "data")
rows = [json.loads(l) for l in open(os.path.join(ROOT, "questions.jsonl"), encoding="utf-8")]
qpaths = {r["path"] for r in rows}
random.seed(0)

for s in sorted(os.listdir(DATA)):
    files = []
    for d, _, fs in os.walk(os.path.join(DATA, s)):
        for f in fs:
            files.append(os.path.relpath(os.path.join(d, f), ROOT).replace("\\", "/"))
    files = [f for f in files if not f.endswith("LICENSE.md")]
    stems = {os.path.splitext(f)[0] for f in files}
    withq = [f for f in files if f in qpaths]
    pat = collections.Counter(re.sub(r"\d+", "#", os.path.basename(f)) for f in files)
    print(f"\n######## {s}: files={len(files)} unique stems={len(stems)} files_with_q={len(withq)}")
    print("name patterns:", pat.most_common(12))
    print("samples:", random.sample(files, min(8, len(files))))
