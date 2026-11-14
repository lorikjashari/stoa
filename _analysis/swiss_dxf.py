"""Parse a swiss_dwellings DXF into rooms, units, and per-layer entity groups."""
import sys, collections


def read_entities(path):
    L = [x.strip() for x in open(path, encoding="utf-8", errors="replace").read().split("\n")]
    s = next(i for i in range(len(L) - 2) if L[i] == "SECTION" and L[i + 2] == "ENTITIES") + 3
    ents, cur = [], None
    for i in range(s, len(L) - 1, 2):
        code, val = L[i], L[i + 1]
        if code == "0":
            if val == "ENDSEC":
                break
            cur = {"type": val, "layer": "", "pts": [], "codes": []}
            ents.append(cur)
        elif cur is not None:
            cur["codes"].append((code, val))
            if code == "8":
                cur["layer"] = val
            elif code == "1":
                cur["text"] = val
            elif code == "10":
                cur["pts"].append([float(val), None])
            elif code == "20" and cur["pts"] and cur["pts"][-1][1] is None:
                cur["pts"][-1][1] = float(val)
            elif code == "70":
                cur["flags"] = int(val)
            elif code == "40":
                cur.setdefault("r", float(val))
    return ents


def area(pts):
    a = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        a += x1 * y2 - x2 * y1
    return abs(a) / 2


def inside(pt, poly):
    x, y = pt
    c = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c


def parse(path):
    ents = read_entities(path)
    outlines = [e["pts"] for e in ents if e["layer"] == "ROOM_OUTLINE" and e["type"] == "LWPOLYLINE"]
    units = [e["pts"] for e in ents if e["layer"] == "UNIT_OUTLINE" and e["type"] == "LWPOLYLINE"]
    texts = [(e["text"], tuple(e["pts"][0])) for e in ents if e["layer"] == "ROOM_TEXT" and e["type"] == "TEXT"]
    outlines = [p for p in outlines if len(p) >= 3 and area(p) > 0]
    rooms = []
    for i, poly in enumerate(outlines):
        a = area(poly)
        holes = [j for j, o in enumerate(outlines) if j != i and area(o) < a and all(inside(p, poly) for p in o)]
        rooms.append({"poly": poly, "gross": a, "holes": holes})
    # a room's text belongs to the smallest polygon containing it
    for s, p in texts:
        cands = [r for r in rooms if inside(p, r["poly"])]
        if cands:
            min(cands, key=lambda r: r["gross"]).setdefault("texts", []).append(s)
    for r in rooms:
        r.setdefault("texts", [])
        # only subtract direct holes (not holes of holes)
        direct = [j for j in r["holes"] if not any(j in rooms[k]["holes"] for k in r["holes"])]
        r["area"] = r["gross"] - sum(rooms[j]["gross"] for j in direct)
    return ents, rooms, units, texts


if __name__ == "__main__":
    ents, rooms, units, texts = parse(sys.argv[1])
    for r in rooms:
        print(round(r["area"], 2), r["texts"])
    print("units:", [round(area(u), 2) for u in units])
    print(collections.Counter((e["layer"], e["type"]) for e in ents))
