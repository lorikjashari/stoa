import json, os, collections
import swiss_dxf as S
from swiss_fixtures import data, ROOT


def dims(pts):
    xs = [x for x, _ in pts]; ys = [y for _, y in pts]
    return max(xs) - min(xs), max(ys) - min(ys)


def ellipse_info(ents):
    rects = [e["pts"] for e in ents if e["layer"] == "FIXTURE" and e["type"] == "LWPOLYLINE" and len(e["pts"]) == 4 and e.get("flags", 0) & 1]
    small = [tuple(e["pts"][0]) + (e["r"],) for e in ents if e["layer"] == "FIXTURE" and e["type"] == "CIRCLE" and e["r"] < 0.035]
    out = []
    for e in ents:
        if e["layer"] == "FIXTURE" and e["type"] == "LWPOLYLINE" and len(e["pts"]) == 48:
            w, h = dims(e["pts"])
            cx = sum(x for x, _ in e["pts"]) / 48; cy = sum(y for _, y in e["pts"]) / 48
            rr = [r for r in rects if S.inside((cx, cy), r)]
            rd = sorted(dims(rr[0])) if rr else None
            has_circle = any(S.inside((x, y), e["pts"]) for x, y, _ in small)
            out.append({"major": max(w, h), "minor": min(w, h), "rect": rd, "circle": has_circle})
    return out


if __name__ == "__main__":
    wc, basin = [], []
    for p, f, a in data:
        if a.get("toilets") is None or a.get("basins") is None:
            continue
        ents = S.read_entities(os.path.join(ROOT, p[:-4] + ".dxf"))
        info = ellipse_info(ents)
        n_c = sum(i["circle"] for i in info)
        if len(info) == a["toilets"] + a["basins"]:
            wc.append(n_c == a["basins"])
    print("ellipses == toilets+basins on", len(wc), "floors; 'ellipse contains small circle' == basin count on", sum(wc))
