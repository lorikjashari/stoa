import os, json, collections

ROOT = r"c:\Users\diar\Desktop\stoa\junction-dataset\junction-dataset"
DATA = os.path.join(ROOT, "data")
out = []
for s in sorted(os.listdir(DATA)):
    rows = 0
    for line in open(os.path.join(DATA, s, "LICENSE.md"), encoding="utf-8"):
        if not line.startswith("| ") or line.startswith("| file") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) < 3:
            continue
        rec = {"source": s, "file": cells[0], "title": cells[1] if len(cells) > 1 else "",
               "licence": cells[2] if len(cells) > 2 else "", "credit": cells[3] if len(cells) > 3 else "",
               "url": cells[4] if len(cells) > 4 else ""}
        out.append(rec)
        rows += 1
    lic = collections.Counter(r["licence"] for r in out if r["source"] == s)
    print(f"{s:16} rows={rows:5} licences={lic.most_common(8)}")

json.dump(out, open(os.path.join(os.path.dirname(__file__), "license_index.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print("header sample opening_design:", [r for r in out if r["source"] == "opening_design"][:3])
print("header sample ifc_models:", [r for r in out if r["source"] == "ifc_models"][:3])
