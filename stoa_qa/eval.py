"""Evaluation harness against Stoa's questions.jsonl.

Scoring follows the dataset README: counts exact; area/length/volume within 2%;
text is free-form (reported as 'unscored' until we add an LLM judge).

    uv run python -m stoa_qa.eval --source swiss_dwellings --per-file 0 --limit 3000
    uv run python -m stoa_qa.eval --sample 2000         # stratified across sources/types
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import time
from pathlib import Path

from .core import DATASET, source_of
from .router import answer


def correct(row: dict, pred) -> bool | None:
    if row["type"] in ("text", "other"):
        return None
    if pred is None or isinstance(pred, str):
        return False
    gt = row["answer"]
    if row["type"] == "count":
        return round(pred) == gt
    return gt != 0 and abs(pred - gt) / abs(gt) <= 0.02


def lang_of(q: str) -> str:
    return "en" if q.isascii() and q.split()[0] in {"How", "What", "Which", "On"} else "other"


def load(source: str | None) -> list[dict]:
    rows = [json.loads(l) for l in open(DATASET / "questions.jsonl", encoding="utf-8")]
    return [r for r in rows if source is None or source_of(r["path"]) == source]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source")
    ap.add_argument("--sample", type=int, help="stratified sample size (by source x type)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--cubicasa-split", choices=["train", "val", "test"],
                    help="keep only cubicasa5k plans from this CubiCasa split (test = held-out, the fair number)")
    args = ap.parse_args()

    rows = load(args.source)
    if args.cubicasa_split:
        from .cubicasa_gemini import split_members
        members = split_members(args.cubicasa_split)
        rows = [r for r in rows if source_of(r["path"]) != "cubicasa5k" or r["path"] in members]
    rnd = random.Random(args.seed)
    if args.sample:
        strata = collections.defaultdict(list)
        for r in rows:
            strata[(source_of(r["path"]), r["type"])].append(r)
        per = max(1, args.sample // len(strata))
        rows = [x for s in strata.values() for x in rnd.sample(s, min(per, len(s)))]
    # group by file so each drawing is parsed once (cache hit for every other question)
    rows.sort(key=lambda r: r["path"])
    if args.limit:
        files = sorted({r["path"] for r in rows})
        rnd.shuffle(files)
        keep = set(files[: max(1, args.limit)])
        rows = [r for r in rows if r["path"] in keep]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = collections.defaultdict(lambda: [0, 0, 0])  # ok, scored, answered
    fails = collections.defaultdict(list)
    t0 = time.time()
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as fp:
        for i, r in enumerate(rows):
            a = answer(r["path"], r["question"])
            ok = correct(r, a.value)
            key = (source_of(r["path"]), r["type"], a.intent)
            s = stats[key]
            if ok is not None:
                s[1] += 1
                s[0] += ok
            s[2] += a.value is not None
            if ok is False:
                fails[key].append({"path": r["path"], "q": r["question"], "gt": r["answer"], "pred": a.value, "method": a.method})
            fp.write(json.dumps({**r, "pred": a.value, "ok": ok, "intent": a.intent, "conf": a.confidence,
                                 "method": a.method, "n_evidence": len(a.evidence)}, ensure_ascii=False) + "\n")
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)

    dt = time.time() - t0
    by_src = collections.defaultdict(lambda: [0, 0])
    print(f"\n{'source':16} {'type':7} {'intent':22} {'acc':>8} {'n':>6}")
    for (src, ty, intent), (ok, n, ans) in sorted(stats.items()):
        by_src[src][0] += ok
        by_src[src][1] += n
        acc = f"{ok / n:.1%}" if n else "-"
        print(f"{src:16} {ty:7} {intent:22} {acc:>8} {n:>6}")
    print()
    tot_ok = sum(v[0] for v in by_src.values())
    tot_n = sum(v[1] for v in by_src.values())
    for src, (ok, n) in sorted(by_src.items()):
        print(f"{src:16} {ok}/{n} = {ok / n:.2%}" if n else src)
    if tot_n:
        print(f"{'TOTAL scored':16} {tot_ok}/{tot_n} = {tot_ok / tot_n:.2%}   ({len(rows)} questions in {dt:.0f}s, {1000 * dt / max(1, len(rows)):.1f} ms/q)")
    with open(out / "failures.json", "w", encoding="utf-8") as fp:
        json.dump({" | ".join(k): v[:50] for k, v in fails.items()}, fp, ensure_ascii=False, indent=1)
    print(f"predictions -> {out}/predictions.jsonl, failures -> {out}/failures.json")


if __name__ == "__main__":
    main()
