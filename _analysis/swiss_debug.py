import sys, json, os, re
import swiss_dxf as S

ROOT = r"c:\Users\diar\Desktop\stoa\junction-dataset\junction-dataset"
stem = sys.argv[1]
path = os.path.join(ROOT, "data", "swiss_dwellings", stem + ".dxf")
ents, rooms, units, texts = S.parse(path)
for i, r in enumerate(rooms):
    c = (sum(x for x, _ in r["poly"]) / len(r["poly"]), sum(y for _, y in r["poly"]) / len(r["poly"]))
    holes = [j for j, o in enumerate(rooms) if j != i and o["area"] < r["area"] and o["poly"] and S.inside(o["poly"][0], r["poly"]) and all(S.inside(p, r["poly"]) for p in o["poly"])]
    print(i, round(r["area"], 2), r["texts"], "npts", len(r["poly"]), "contains", holes)
print("units", [round(S.area(u), 2) for u in units])
print("texts", texts)
for e in ents:
    if e["layer"] == "ROOM_TEXT":
        print(e["text"], e["codes"][:30]); break
for l in open(os.path.join(ROOT, "questions.jsonl"), encoding="utf-8"):
    r = json.loads(l)
    if stem + "." in r["path"] and r["question"].startswith(("How", "What")):
        print(r["question"], "=>", r["answer"])
