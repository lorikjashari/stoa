"""Geometry vs Stoa answers on the val split: what post-processing could recover."""
import collections
import json
from pathlib import Path

from shapely.geometry import Polygon

from stoa_qa.cubicasa_gemini import load_truth, split_members

REPO = Path(__file__).resolve().parent.parent
truth = load_truth()
preds = {}
for f in (REPO / "third_party/cubicasa_preds").glob("*.json"):
    r = json.loads(f.read_text())
    if "counts" in r and "rooms" in r:
        preds[r["path"]] = r

val = sorted(p for p in split_members("val") if p in truth and p in preds)
print("val cached", len(val))


def poly(pts):
    if len(pts) < 3:
        return None
    g = Polygon(pts)
    if not g.is_valid:
        g = g.buffer(0)
    return g if g.area > 0 else None


def inside(room, icon, min_frac=0.5):
    rp, ip = poly(room["poly"]), poly(icon["poly"])
    if rp is None or ip is None or ip.area == 0:
        return False
    return rp.intersects(ip) and rp.intersection(ip).area / ip.area >= min_frac


# Bath rooms: icons inside vs truth bathrooms / wc
rows = []
for p in val:
    t = truth[p]
    if "bathrooms" not in t and "wc_rooms" not in t:
        continue
    pr = preds[p]
    baths = [r for r in pr["rooms"] if r["class"] == "Bath"]
    icons = pr.get("icons", [])
    info = []
    for r in baths:
        kinds = collections.Counter(i["class"] for i in icons if inside(r, i))
        area = poly(r["poly"]).area if poly(r["poly"]) else 0
        info.append((area, kinds))
    rows.append((t.get("bathrooms"), t.get("wc_rooms"), t.get("baths"), pr["counts"].get("baths"), info))

print("\nplans with bath truth", len(rows))
# predict wc = bath rooms with toilet and no bathtub; bathroom = the rest of bath rooms
ok_wc = ok_bath = n_wc = n_bath = 0
for tbath, twc, tbaths, pbaths, info in rows:
    wc = sum(1 for area, k in info if k["Toilet"] and not k["Bathtub"])
    bath = len(info) - wc
    if twc is not None:
        n_wc += 1
        ok_wc += wc == twc
    if tbath is not None:
        n_bath += 1
        ok_bath += bath == tbath
print(f"wc = bathroom+toilet-no-tub: {ok_wc}/{n_wc} = {ok_wc / max(n_wc,1):.1%}")
print(f"bathroom = other bath rooms: {ok_bath}/{n_bath} = {ok_bath / max(n_bath,1):.1%}")

# showers = bath rooms with no bathtub
ok = n = 0
for tbath, twc, tbaths, pbaths, info in rows:
    if tbaths is None:
        continue
    showers = sum(1 for area, k in info if not k["Bathtub"])
    pred = (pbaths or 0) + showers
    n += 1
    ok += pred == tbaths
print(f"baths = tubs + bath rooms without a tub: {ok}/{n} = {ok / max(n,1):.1%}")

# area split: WC rooms are smaller. sweep threshold on bath-room area
print("\narea threshold for WC (rooms with toilet, no tub, area < T counted as WC; rest of bath rooms = bathroom)")
for T in (2000, 4000, 6000, 8000, 12000, 20000, 10**9):
    ok_wc = ok_bath = n_wc = n_bath = 0
    for tbath, twc, tbaths, pbaths, info in rows:
        wc = sum(1 for area, k in info if k["Toilet"] and not k["Bathtub"] and area < T)
        bath = len(info) - wc
        if twc is not None:
            n_wc += 1
            ok_wc += wc == twc
        if tbath is not None:
            n_bath += 1
            ok_bath += bath == tbath
    print(f"  T={T:<8} wc {ok_wc/max(n_wc,1):.1%} ({n_wc})  bath {ok_bath/max(n_bath,1):.1%} ({n_bath})")

# outdoor area vs balconies
print("\noutdoor rooms vs balconies")
ok = n = 0
base_ok = base_n = 0
for p in val:
    if "balconies" not in truth[p]:
        continue
    rooms = [r for r in preds[p]["rooms"] if r["class"] == "Outdoor"]
    base_n += 1
    base_ok += len(rooms) == truth[p]["balconies"]
print(f"raw outdoor count {base_ok}/{base_n} = {base_ok/max(base_n,1):.1%}")
for T in (3000, 5000, 8000, 12000, 20000, 40000):
    ok = n = 0
    for p in val:
        if "balconies" not in truth[p]:
            continue
        rooms = [r for r in preds[p]["rooms"] if r["class"] == "Outdoor"]
        pred = sum(1 for r in rooms if poly(r["poly"]) and poly(r["poly"]).area < T)
        n += 1
        ok += pred == truth[p]["balconies"]
    print(f"  outdoor area < {T}: {ok}/{n} = {ok/max(n,1):.1%}")

# door polygon widths
print("\ndoor width vs count (sample of deltas)")
deltas = collections.Counter()
for p in val[:]:
    if "doors" not in truth[p]:
        continue
    doors = [i for i in preds[p].get("icons", []) if i["class"] == "Door"]
    deltas[int(preds[p]["counts"]["doors"]) - truth[p]["doors"]] += 1
print("door pred-truth", sorted(deltas.items())[:20], "n", sum(deltas.values()))
wdelta = collections.Counter()
for p in val:
    if "windows" not in truth[p]:
        continue
    wdelta[int(preds[p]["counts"]["windows"]) - truth[p]["windows"]] += 1
print("window pred-truth", sorted(wdelta.items())[:24], "n", sum(wdelta.values()))
