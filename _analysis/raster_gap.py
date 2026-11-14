"""Error profile of cached CubiCasa network predictions vs Stoa's answers."""
import collections
import json
from pathlib import Path

from stoa_qa.cubicasa_gemini import QUESTIONS, load_truth, split_members

REPO = Path(__file__).resolve().parent.parent
truth = load_truth()
preds = {}
for f in (REPO / "third_party/cubicasa_preds").glob("*.json"):
    r = json.loads(f.read_text())
    if "counts" in r:
        preds[r["path"]] = r

KEYS = [k for k, _, _ in QUESTIONS]
for split in ("val", "test"):
    members = sorted(p for p in split_members(split) if p in truth)
    have = [p for p in members if p in preds]
    print(f"=== {split}: plans {len(members)} cached {len(have)} ===")
    floors = collections.Counter(truth[p].get("floors") for p in members)
    print(" floors truth", dict(sorted((k, v) for k, v in floors.items() if k is not None)))
    header = f"{'key':12} {'n':>5} {'exact':>7} {'mae':>6} {'under':>6} {'over':>6} {'missed':>7}"
    print(header)
    for k in KEYS:
        ok = n = under = over = zp = ae = 0
        for p in have:
            if k not in truth[p] or k not in preds[p]["counts"]:
                continue
            g = truth[p][k]
            pr = int(preds[p]["counts"][k])
            n += 1
            ok += pr == g
            ae += abs(pr - g)
            under += pr < g
            over += pr > g
            zp += pr == 0 and g > 0
        if n:
            print(f"{k:12} {n:5} {ok / n:7.1%} {ae / n:6.2f} {under / n:6.1%} {over / n:6.1%} {zp / n:7.1%}")
        else:
            print(f"{k:12} no network count")
