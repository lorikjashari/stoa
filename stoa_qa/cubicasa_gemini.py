"""Frontier-model-only baseline on cubicasa5k: send the plan image to Gemini, ask all 12 questions at once.

This is the comparison Stoa asks for ("send the image to a frontier LLM and ask"). It is scored on the
same held-out CubiCasa test plans as blueprint3d/eval_cubicasa_model.py, so the numbers line up.

    uv run python -m stoa_qa.cubicasa_gemini --model gemini-3.5-flash --limit 100
    uv run python -m stoa_qa.cubicasa_gemini --model gemini-3.1-pro-preview --limit 40

One request per plan answers all 12 questions (English and Finnish rows share the answer), so the cost
per question is the per-plan cost / 24. Responses are cached in eval_out/gemini/<model>/<plan>.json.
"""
from __future__ import annotations

import argparse
import base64
import collections
import io
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import dotenv_values
from PIL import Image

from .core import DATASET

REPO = Path(__file__).resolve().parent.parent
SPLITS = REPO / "third_party/cubicasa_split"

# key, English question prefix in questions.jsonl, definition given to the model (taken from the questions)
QUESTIONS = [
    ("doors", "How many doors", "doorways fitted with a door leaf; open doorways without a door leaf do not count"),
    ("windows", "How many windows", "windows drawn on the plan"),
    ("floors", "How many floors", "number of floors (storeys) the plan shows"),
    ("toilets", "How many toilet bowls", "toilet bowls (WC seats) drawn"),
    ("sinks", "How many sinks and wash basins", "sinks and wash basins, kitchen and bathroom combined; a double-bowl sink counts as one"),
    ("baths", "How many showers and bathtubs", "showers and bathtubs drawn, in total"),
    ("saunas", "How many saunas", "saunas (rooms marked S or SAUNA)"),
    ("kitchens", "How many kitchens", "kitchens, including kitchenettes, open kitchens and utility kitchens (KEITTIÖ, KK, TK, AVK...)"),
    ("bedrooms", "How many bedrooms", "bedrooms (makuuhuone, usually marked MH)"),
    ("wc_rooms", "How many separate WC rooms", "separate WC rooms (toilet rooms marked WC)"),
    ("bathrooms", "How many bathrooms or washrooms", "bathrooms or washrooms (KH, KPH, PH, PESUH, SH), not counting separate WC rooms"),
    ("balconies", "How many balconies", "balconies (parveke, including glazed balconies)"),
]

PROMPT = """You are reading a Finnish residential floor plan image. Count each item below exactly as defined.
Count carefully, element by element. If an item does not appear, answer 0.

{items}

Return JSON only: an object with one integer per key."""

PROMPT_BOXES = """You are reading a Finnish residential floor plan image. Find every instance of each item below,
exactly as defined, and give each one a tight bounding box as [ymin, xmin, ymax, xmax] normalised to 0-1000.
List each physical item once. For rooms, box the whole room. If an item does not appear, return an empty list.

{items}

Return JSON only: an object with one list of boxes per key."""


def split_members(split: str) -> set[str]:
    rows = (SPLITS / f"cubicasa5k__{split}.txt").read_text().split()
    return {f"data/cubicasa5k/{r.strip('/').replace('/', '-')}.png" for r in rows}


def load_truth() -> dict[str, dict[str, int]]:
    truth: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for line in open(DATASET / "questions.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if "/cubicasa5k/" in r["path"]:
            for k, prefix, _ in QUESTIONS:
                if r["question"].startswith(prefix):
                    truth[r["path"]][k] = r["answer"]
    return truth


def image_b64(path: Path, max_side: int) -> str:
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def ask(model: str, key: str, path: str, max_side: int, cache_dir: Path, boxes: bool = False) -> dict:
    out = cache_dir / (Path(path).stem + ".json")
    if out.exists():
        return json.loads(out.read_text())
    items = "\n".join(f'- "{k}": {d}' for k, _, d in QUESTIONS)
    box_list = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "box_2d": {"type": "ARRAY", "items": {"type": "INTEGER"}}}, "required": ["box_2d"]}}
    value_schema = box_list if boxes else {"type": "INTEGER"}
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/png", "data": image_b64(DATASET / path, max_side)}},
            {"text": (PROMPT_BOXES if boxes else PROMPT).format(items=items)},
        ]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {"type": "OBJECT", "properties": {k: value_schema for k, _, _ in QUESTIONS},
                               "required": [k for k, _, _ in QUESTIONS]},
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    t0 = time.time()
    for attempt in range(5):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < 4:
                time.sleep(2 ** attempt * 5)
                continue
            return {"path": path, "error": f"HTTP {e.code}: {e.read()[:200]!r}"}
        except Exception as e:  # network hiccup
            if attempt < 4:
                time.sleep(5)
                continue
            return {"path": path, "error": repr(e)[:200]}
    try:
        parsed = json.loads(resp["candidates"][0]["content"]["parts"][0]["text"])
    except Exception as e:
        return {"path": path, "error": f"bad response: {e!r} {str(resp)[:200]}"}
    # "floors" is not a boxable object: in box mode the model lists one box per drawn storey
    counts = {k: len(v) for k, v in parsed.items()} if boxes else parsed
    res = {"path": path, "model": model, "seconds": round(time.time() - t0, 1), "counts": counts,
           "boxes": parsed if boxes else None, "usage": resp.get("usageMetadata", {})}
    out.write_text(json.dumps(res))
    return res


def ask_featherless(model: str, key: str, path: str, max_side: int, cache_dir: Path) -> dict:
    """Same prompt through Featherless (OpenAI-compatible), e.g. google/gemma-4-31B-it."""
    out = cache_dir / (Path(path).stem + ".json")
    if out.exists():
        return json.loads(out.read_text())
    items = "\n".join(f'- "{k}": {d}' for k, _, d in QUESTIONS)
    body = {"model": model, "max_tokens": 600, "temperature": 0, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64(DATASET / path, max_side)}},
        {"type": "text", "text": PROMPT.format(items=items)}]}]}
    t0 = time.time()
    for attempt in range(6):
        req = urllib.request.Request("https://api.featherless.ai/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 5:
                time.sleep(10 * (attempt + 1))
                continue
            return {"path": path, "error": f"HTTP {e.code}: {e.read()[:200]!r}"}
        except Exception as e:
            if attempt < 5:
                time.sleep(10)
                continue
            return {"path": path, "error": repr(e)[:200]}
    text = resp["choices"][0]["message"]["content"]
    try:
        counts = json.loads(text[text.index("{"): text.rindex("}") + 1])
    except Exception as e:
        return {"path": path, "error": f"bad response: {e!r} {text[:200]}"}
    u = resp.get("usage", {})
    res = {"path": path, "model": model, "seconds": round(time.time() - t0, 1), "counts": counts,
           "usage": {"promptTokenCount": u.get("prompt_tokens", 0), "candidatesTokenCount": u.get("completion_tokens", 0),
                     "cost_usd": u.get("cost")}}
    out.write_text(json.dumps(res))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=2048)
    ap.add_argument("--provider", default="gemini", choices=["gemini", "featherless"])
    ap.add_argument("--boxes", action="store_true", help="list every item with a box (evidence); count = list length")
    a = ap.parse_args()

    env_key = "GEMINI_API_KEY" if a.provider == "gemini" else "FEATHERLESS_API_KEY"
    key = os.environ.get(env_key) or dotenv_values(REPO / ".env").get(env_key)
    if not key:
        raise SystemExit(f"{env_key} missing (set it in .env)")
    truth = load_truth()
    plans = sorted(p for p in split_members(a.split) if p in truth)[: a.limit]
    cache = REPO / "eval_out/gemini" / (a.model.replace("/", "__") + ("-boxes" if a.boxes else ""))
    cache.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        if a.provider == "featherless":
            results = list(ex.map(lambda p: ask_featherless(a.model, key, p, a.max_side, cache), plans))
        else:
            results = list(ex.map(lambda p: ask(a.model, key, p, a.max_side, cache, a.boxes), plans))
    errors = [r for r in results if "error" in r]

    stats = collections.defaultdict(lambda: [0, 0, 0])  # exact, within 1, n
    tok_in = tok_out = 0
    cost = 0.0
    for r in results:
        if "error" in r:
            continue
        u = r.get("usage", {})
        tok_in += u.get("promptTokenCount", 0)
        tok_out += u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
        cost += u.get("cost_usd") or 0.0
        for k, _, _ in QUESTIONS:
            if k in truth[r["path"]] and k in r["counts"]:
                s, pred, gold = stats[k], int(r["counts"][k]), truth[r["path"]][k]
                s[0] += pred == gold
                s[1] += abs(pred - gold) <= 1
                s[2] += 1

    ok = len(results) - len(errors)
    print(f"model={a.model}{' (boxes)' if a.boxes else ''} split={a.split}: {len(plans)} plans, {ok} answered, {len(errors)} errors, {time.time() - t0:.0f}s")
    for r in errors[:3]:
        print("  error:", r["path"], r["error"])
    print(f"\n{'question':<12}{'exact':>8}{'±1':>8}{'n':>6}")
    E = N = 0
    for k, _, _ in QUESTIONS:
        e, w1, n = stats[k]
        if n:
            print(f"{k:<12}{e / n:>8.1%}{w1 / n:>8.1%}{n:>6}")
            E, N = E + e, N + n
    if N:
        print(f"{'ALL':<12}{E / N:>8.1%}{'':>8}{N:>6}")
    if ok:
        print(f"\ntokens per plan: {tok_in / ok:.0f} in, {tok_out / ok:.0f} out (incl. thinking); "
              f"one request answers ~24 questions (12 EN + 12 FI)")
        if cost:
            print(f"cost: ${cost / ok:.5f} per plan, ${cost / ok / 24 * 1000:.3f} per 1,000 questions")


if __name__ == "__main__":
    main()
