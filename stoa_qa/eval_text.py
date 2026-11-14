"""Free-form (natural-language) question evaluation: the full answer pipeline, graded against Stoa's reference text.

Stoa's `text` questions (ifc_bench: 820 IFC-Bench v2 questions; opening_design: 16 schedule look-ups) have
prose reference answers, so exact matching is impossible. Every question runs through stoa_qa.pipeline.solve
exactly as the app answers it, then an LLM grader compares the answer with the reference:

    correct    every fact the reference asks for is present and right (numbers within 2% or rounding,
               units converted are fine); "not in the model" answered as not available counts as correct
    partial    some of the requested facts are right, others missing or wrong
    incorrect  wrong, missing, or "can't answer" where the reference has the facts

    uv run python -m stoa_qa.eval_text --limit 40          # pilot on 40 random questions
    uv run python -m stoa_qa.eval_text                      # everything (resumable: answered rows are skipped)

The grader is the same Gemma 4 model the planner uses, so its grades are spot-checked by hand in the report.
Writes eval_out/text_eval/{predictions.jsonl, summary.json}.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import catalog, llm, pipeline
from .core import DATASET, source_of

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "eval_out" / "text_eval"

GRADER = (
    "You grade answers about building models and construction drawings against a reference answer written by an expert. "
    "Reply with one JSON object: {\"verdict\": \"correct\" | \"partial\" | \"incorrect\", \"reason\": \"<one short sentence>\"}.\n"
    "- correct: every fact the question asks for is present and agrees with the reference. Numbers may differ by up to 2% "
    "or by rounding; unit conversions (sq ft vs m2, inches vs m) are fine; extra correct detail is fine; wording and order do not matter.\n"
    "- partial: part of what was asked is right, but some requested facts are missing or wrong.\n"
    "- incorrect: the facts are wrong or missing, or the answer declines although the reference has the information.\n"
    "- If the reference says the information is NOT in the model, an answer that says it is unavailable or cannot be answered is correct, "
    "and an answer that invents a definite value is incorrect.\n"
    "Judge only against the reference, not your own knowledge."
)


def rows_for(sources: list[str]) -> list[dict]:
    rows = []
    for line in open(DATASET / "questions.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["type"] in ("text", "other") and source_of(r["path"]) in sources:
            rows.append(r)
    return rows


def key_of(r: dict) -> str:
    return f"{r['path']}|{r['question']}"


def grade(question: str, reference: str, answer: str) -> tuple[dict, llm.Usage | None]:
    user = f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\nANSWER TO GRADE:\n{answer}"
    try:
        g, u = llm.chat_json(GRADER, user, max_tokens=200)
    except llm.LLMError as e:
        return {"verdict": "error", "reason": str(e)[:200]}, None
    v = str(g.get("verdict", "")).lower().strip()
    return {"verdict": v if v in ("correct", "partial", "incorrect") else "error", "reason": str(g.get("reason", ""))[:300]}, u


def run_one(d: catalog.Drawing, r: dict) -> dict:
    t0 = time.time()
    try:
        res = pipeline.solve(d, r["question"])
        a = res.answer
        text = pipeline.answer_text(a)
        engine, usage = res.engine, list(res.usage)
    except Exception as e:  # never stop the batch
        text, engine, usage, a = f"(pipeline error: {type(e).__name__}: {e})", "error", [], None
    seconds = time.time() - t0
    g, gu = grade(r["question"], str(r["answer"]), text)
    return {"path": r["path"], "question": r["question"], "reference": r["answer"], "answer": text,
            "method": (a.method if a else "")[:600], "n_evidence": len(a.evidence) if a else 0,
            "engine": engine, "seconds": round(seconds, 2),
            "tokens": sum(u["tokens_in"] + u["tokens_out"] for u in usage), "cost_usd": round(sum(u["cost_usd"] for u in usage), 6),
            "grade": g["verdict"], "grade_reason": g["reason"],
            "grade_cost_usd": round(gu.cost_usd, 6) if gu else 0.0}


def summarise(recs: list[dict]) -> dict:
    by = collections.defaultdict(collections.Counter)
    for x in recs:
        by[source_of(x["path"])][x["grade"]] += 1
        by["ALL"][x["grade"]] += 1
    eng = collections.defaultdict(collections.Counter)
    for x in recs:
        eng[x["engine"]][x["grade"]] += 1
    n = len(recs)
    return {
        "questions": n,
        "by_source": {s: dict(c) for s, c in by.items()},
        "by_engine": {e: dict(c) for e, c in eng.items()},
        "correct_rate": round(by["ALL"]["correct"] / max(1, n), 4),
        "correct_or_partial_rate": round((by["ALL"]["correct"] + by["ALL"]["partial"]) / max(1, n), 4),
        "answered_rate": round(sum(not x["answer"].startswith("I can't") for x in recs) / max(1, n), 4),
        "median_seconds": sorted(x["seconds"] for x in recs)[n // 2] if n else None,
        "answer_cost_usd": round(sum(x["cost_usd"] for x in recs), 4),
        "grade_cost_usd": round(sum(x["grade_cost_usd"] for x in recs), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["ifc_bench", "opening_design"])
    ap.add_argument("--limit", type=int, help="random sample of this many questions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="concurrent LLM calls (per drawing)")
    ap.add_argument("--out", default=None, help="output folder (default eval_out/text_eval)")
    ap.add_argument("--redo-unanswered", action="store_true",
                    help="re-run only the questions the default run answered with \"I can't answer\" (into --out)")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)
    if not llm.available():
        raise SystemExit("FEATHERLESS_API_KEY is not set: the planner and the grader both need it")

    rows = rows_for(a.sources)
    if a.redo_unanswered:
        prev = REPO / "eval_out" / "text_eval" / "predictions.jsonl"
        cant = {key_of(x) for x in map(json.loads, prev.read_text(encoding="utf-8").splitlines()) if x["answer"].startswith("I can't")}
        rows = [r for r in rows if key_of(r) in cant]
    if a.limit:
        rows = random.Random(a.seed).sample(rows, min(a.limit, len(rows)))
    OUT.mkdir(parents=True, exist_ok=True)
    pred_file = OUT / "predictions.jsonl"
    done = {}
    if pred_file.exists():
        for line in pred_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                x = json.loads(line)
                if x["grade"] != "error":
                    done[key_of(x)] = x
    todo = [r for r in rows if key_of(r) not in done]
    files = {f: d for d in catalog.catalog().values() for f in d.files}
    by_path = collections.defaultdict(list)
    for r in todo:
        by_path[r["path"]].append(r)
    print(f"{len(rows)} questions, {len(done)} already graded, {len(todo)} to run over {len(by_path)} drawings", flush=True)
    t0 = time.time()
    with open(pred_file, "a", encoding="utf-8") as fp:
        for i, (path, rs) in enumerate(sorted(by_path.items())):
            d = files.get(path)
            if d is None:
                print(f"  no drawing for {path}", flush=True)
                continue
            from . import planner

            planner.tables_for(d.ask_path)  # parse the model once before the threads share it
            with ThreadPoolExecutor(a.workers) as pool:
                for rec in pool.map(lambda r: run_one(d, r), rs):
                    done[key_of(rec)] = rec
                    fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fp.flush()
            c = collections.Counter(x["grade"] for x in done.values() if key_of(x) in {key_of(r) for r in rows})
            print(f"  [{i + 1}/{len(by_path)}] {Path(path).name}: {len(rs)} q  | so far {dict(c)}  {time.time() - t0:.0f}s", flush=True)
    recs = [done[key_of(r)] for r in rows if key_of(r) in done]
    s = summarise(recs)
    (OUT / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
