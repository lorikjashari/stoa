"""Probe the app's answer pipeline with realistic user questions on every kind of drawing.

Not Stoa's benchmark: these are the free-form questions a site engineer would type, including ones the data cannot
answer, to see how often the app answers, how it explains a refusal, and what it costs.

    venv/Scripts/python.exe _analysis/app_probe.py [out.json]     # writes eval_out/probe/<out>.json
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from stoa_qa import api, pipeline  # noqa: E402

GENERIC = ["What is the scale of this drawing?", "Who is the architect or author of this drawing?", "What is the ceiling height?",
           "What are the walls made of?", "Which direction is north?", "Is there an elevator?"]
CASES = {
    "data/swiss_dwellings/site1000_bldg2082_floor9385.pdf": [
        "How many bedrooms are there?", "What is the total living area?", "Which apartment is the biggest?",
        "How many bathrooms does apartment B have?", "Is the kitchen open to the living room?", "How wide is the corridor?",
        "How many rooms have a balcony access?", "What is the area of the smallest room?"] + GENERIC,
    "data/ifc_models/kit-fzk-haus_ac20-fzk-haus.ifc": [
        "How many rooms are on the ground floor?", "What is the total floor area of the house?", "How many windows face south?",
        "What is the roof type?", "How tall is the building?", "What is the area of the kitchen?", "What materials are used in the walls?",
        "How many doors lead outside?"] + GENERIC,
    "data/cubicasa5k/high_quality_architectural-1834.png": [
        "How many bedrooms are there?", "How many bathrooms?", "Is there a sauna?", "What is the total area of the apartment?",
        "How many balconies?", "Which room is the largest?", "How many doors are there?"] + GENERIC,
    "data/opening_design/Sports_Complex/Sheets - Current Set/A-801 - Door and Finish Schedules & Wall Types.pdf": [
        "How many doors are in the door schedule?", "What is the fire rating of door 101?", "Which rooms have carpet flooring?",
        "What wall types are used?", "How many doors are 3'-0\" wide?"] + GENERIC,
    "data/habs/ak0003_sheet_00019.jpg": ["What building is this?", "How many floors does the building have?", "How many windows are on the front facade?",
                                         "When was this drawing made?"] + GENERIC[:3],
    "data/wikimedia/100075564.jpg": ["How many rooms are there?", "What kind of building is this?", "How many doors are shown?"] + GENERIC[:2],
}


def main() -> None:
    out_name = sys.argv[1] if len(sys.argv) > 1 else "before"
    rows = []
    for did, qs in CASES.items():
        try:
            d = api._drawing(did)
        except Exception as e:  # file not in this copy of the dataset
            print("skip", did, e)
            continue
        for q in qs:
            t = time.perf_counter()
            r = pipeline.respond(d, q)
            rows.append({"drawing": did, "source": d.source, "q": q, "text": r["text"], "value": r["value"] is not None,
                         "engine": r["engine"]["kind"], "method": r["method"], "conf": r["confidence"],
                         "tokens": r["engine"]["llm_tokens"], "cost": r["engine"]["cost_usd"], "s": round(time.perf_counter() - t, 1),
                         "crew": [(c["agent"], c["outcome"]) for c in r.get("crew") or []], "explain": r.get("unanswered")})
            print(f"{'OK ' if rows[-1]['value'] else '-- '}{d.source[:10]:10} {r['engine']['kind']:9} {q[:55]:55} | {r['text'][:90]!r}", flush=True)
    dest = REPO / "eval_out" / "probe" / f"{out_name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    n = len(rows)
    print(f"\nanswered {sum(r['value'] for r in rows)}/{n}; tokens {sum(r['tokens'] for r in rows)}; cost ${sum(r['cost'] for r in rows):.4f} -> {dest}")


if __name__ == "__main__":
    main()
