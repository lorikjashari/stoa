"""Check how well DXF-derived values reproduce swiss_dwellings ground truth."""
import json, os, re, collections
import swiss_dxf as S

ROOT = r"c:\Users\diar\Desktop\stoa\junction-dataset\junction-dataset"
rows = [json.loads(l) for l in open(os.path.join(ROOT, "questions.jsonl"), encoding="utf-8")]
by_file = collections.defaultdict(list)
for r in rows:
    if "/swiss_dwellings/" in r["path"] and r["question"].isascii() or ("/swiss_dwellings/" in r["path"] and r["question"].startswith(("How", "What"))):
        by_file[r["path"]].append(r)

OUTDOOR = {"Balkon", "Balcon", "Balcone", "Loggia", "Terrasse", "Terrazza", "Sitzplatz", "Jardin d'hiver", "Wintergarten", "Giardino d'inverno"}
import sys
USE_PRINTED = "--printed" in sys.argv
USE_FILL = "--fill" in sys.argv
from swiss_pdf_fill import room_fills
stats = collections.defaultdict(lambda: [0, 0])
printed_floors = 0
fails = collections.defaultdict(list)


def ok(kind, pred, ans, tol=None):
    s = stats[kind]
    s[1] += 1
    good = (pred == ans) if tol is None else (ans != 0 and abs(pred - ans) / abs(ans) <= tol)
    if good:
        s[0] += 1
    return good


for path, qs in by_file.items():
    dxf = os.path.join(ROOT, path[:-4] + ".dxf")
    ents, rooms, units, texts = S.parse(dxf)
    named = []
    for rm in rooms:
        nums = [t for t in rm["texts"] if re.fullmatch(r"U?\d+\.\d+", t)]
        printed = [t for t in rm["texts"] if t.endswith("m²")]
        labels = [t for t in rm["texts"] if t not in nums and t not in printed and not re.match(r"(Whg\.|Appt\.?|App\.)\s*[A-Z]$", t)]
        if nums:
            a = float(printed[0].split()[0]) if printed and USE_PRINTED else rm["area"]
            anchor = next(p for t, p in texts if t == nums[0] and S.inside(p, rm["poly"]))
            named.append({"num": nums[0], "label": labels[0] if labels else None, "area": a, "poly": rm["poly"], "printed": bool(printed), "anchor": anchor})
    printed_floors += any(n["printed"] for n in named)
    if USE_FILL:
        fills = room_fills(os.path.basename(path)[:-4])
        for n in named:
            n["outdoor"] = abs(fills.get(n["num"], 1.0) - 0.894) < 0.01
    else:
        for n in named:
            n["outdoor"] = n["label"] in OUTDOOR
    windows = sum(1 for e in ents if e["layer"] == "WINDOW" and e["type"] == "LWPOLYLINE" and e.get("flags", 0) & 1)
    doors = sum(1 for e in ents if e["layer"] == "DOOR" and e["type"] == "LWPOLYLINE" and len(e["pts"]) == 5)

    def centroid(p):
        return (sum(x for x, _ in p) / len(p), sum(y for _, y in p) / len(p))

    unit_letters = [t for t, _ in texts if re.match(r"(Whg\.|Appt\.?|App\.)\s*[A-Z]", t)]
    for r in qs:
        q, a = r["question"], r["answer"]
        m = re.match(r"What is the net floor area of room (\S+) \('(.+)'\) in m²\?", q)
        if m:
            cand = [n for n in named if n["num"] == m.group(1)]
            pred = cand[0]["area"] if cand else -1
            if not ok("room_area", round(pred, 2), a, 0.02):
                fails["room_area"].append((path, q, a, pred))
            continue
        m = re.match(r"How many rooms labelled '(.+)' are there on this floor\?", q)
        if m:
            pred = sum(1 for n in named if n["label"] == m.group(1))
            if not ok("label_count", pred, a):
                fails["label_count"].append((path, q, a, pred))
            continue
        m = re.match(r"What is the total net area in m² of all rooms labelled '(.+)' on this floor\?", q)
        if m:
            pred = sum(n["area"] for n in named if n["label"] == m.group(1))
            if not ok("label_area", pred, a, 0.02):
                fails["label_area"].append((path, q, a, pred))
            continue
        if q.startswith("What is the total net floor area in m² of all numbered indoor rooms"):
            pred = sum(n["area"] for n in named if not n["outdoor"])
            if not ok("floor_area", pred, a, 0.02):
                fails["floor_area"].append((path, q, a, pred, sorted({str(n["label"]) for n in named})))
            continue
        m = re.match(r"What is the net floor area of apartment ([A-Z]) in m²", q)
        if m:
            letter = m.group(1)
            idx = ord(letter) - 65
            pred = -1
            # find the unit outline containing the "Whg. X" label
            lab = [p for t, p in texts if re.match(r"(Whg\.|Appt\.?|App\.)\s*" + letter + r"$", t)]
            if lab:
                us = [u for u in units if S.inside(lab[0], u)]
                if us:
                    pred = sum(n["area"] for n in named if not n["outdoor"] and S.inside(n["anchor"], us[0]))
            if not ok("apt_area", pred, a, 0.02):
                fails["apt_area"].append((path, q, a, pred))
            continue
        if q.startswith("How many apartments"):
            if not ok("apartments", len(unit_letters), a):
                fails["apartments"].append((path, a, len(units), unit_letters))
            continue
        if q.startswith("How many windows"):
            if not ok("windows", windows, a):
                fails["windows"].append((path, a, windows))
            continue
        if q.startswith("How many doors"):
            if not ok("doors", doors, a):
                fails["doors"].append((path, a, doors))
            continue
        stats["unhandled:" + q[:40]][1] += 1

print("floors with printed areas:", printed_floors)
for k, (g, n) in sorted(stats.items()):
    if k.startswith("unhandled"):
        continue
    print(f"{k:45} {g}/{n} = {g / n:.4f}" if n else k)
for k, v in fails.items():
    print("\n--", k, len(v))
    for x in v[:6]:
        print("  ", x)


