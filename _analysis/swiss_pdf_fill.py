import fitz, re, sys, os

ROOT = r"c:\Users\diar\Desktop\stoa\junction-dataset\junction-dataset\data\swiss_dwellings"


def room_fills(stem):
    d = fitz.open(os.path.join(ROOT, stem + ".pdf"))
    p = d[0]
    rooms = [x for x in p.get_drawings() if x.get("fill") is not None and x.get("color") is not None
             and abs(x["color"][0] - 0.541) < 0.01]
    words = p.get_text("words")
    out = {}
    for w in words:
        t = w[4]
        if re.fullmatch(r"U?\d+\.\d+", t):
            cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
            cands = [r for r in rooms if r["rect"].contains(fitz.Point(cx, cy))]
            if cands:
                r = min(cands, key=lambda r: r["rect"].width * r["rect"].height)
                out[t] = round(r["fill"][0], 3)
    return out


if __name__ == "__main__":
    for k, v in sorted(room_fills(sys.argv[1]).items()):
        print(k, v)
