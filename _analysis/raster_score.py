"""Score cubicasa_refine on one split. Default is val, which is what the thresholds were fit on."""
import json
import sys
from pathlib import Path

from stoa_qa.cubicasa_gemini import load_truth, split_members
from stoa_qa.cubicasa_refine import apply

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    truth = load_truth()
    members = split_members(split)
    seg = {}
    cache = REPO / "eval_out" / "seg_components.jsonl"
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                seg[r["path"]] = r
    rows = []
    for f in (REPO / "third_party" / "cubicasa_preds").glob("*.json"):
        pred = json.loads(f.read_text())
        path = pred.get("path")
        if path in members and path in truth and "rooms" in pred:
            rows.append((path, pred, seg.get(path), truth[path]))
    print(f"{split}: {len(rows)} plans, seg for {sum(p in seg for p, *_ in rows)}")
    keys = ["doors", "windows", "toilets", "sinks", "baths", "floors", "wc_rooms", "bathrooms",
            "saunas", "kitchens", "bedrooms", "balconies"]
    raw = {k: [0, 0] for k in keys}
    new = {k: [0, 0] for k in keys}
    for path, pred, got_seg, gold in rows:
        got = apply(pred, got_seg)
        for k in keys:
            if k not in gold:
                continue
            if k in (pred.get("counts") or {}):
                raw[k][1] += 1
                raw[k][0] += int(pred["counts"][k]) == gold[k]
            new[k][1] += 1
            new[k][0] += got.get(k) == gold[k]
    print(f"{'question':12} {'before':>8} {'after':>8} {'n':>6}")
    ok = n = 0
    for k in keys:
        a, m = new[k]
        b, old_n = raw[k]
        if m:
            before = f"{b / old_n:.1%}" if old_n else "—"
            print(f"{k:12} {before:>8} {a / m:8.1%} {m:6}")
            ok += a
            n += m
    print(f"{'ALL':12} {'':>8} {ok / max(n, 1):8.1%} {n:6}")


if __name__ == "__main__":
    main()
