"""Free-form questions: an LLM plans ONE query over the drawing's fact tables; we execute it exactly.

The model (Gemma 4) never sees or produces the numbers in the answer. It only chooses table, filters
and aggregation; the result and its evidence come from geometry / BIM data. The plan itself is shown
to the user, so the reasoning can be checked as well as the highlighted elements.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

from . import graph, ifc, ifc_props, llm, mep, swiss
from .core import DATASET, Answer, fmt_value, source_of

OPS = {"=", "!=", ">", ">=", "<", "<=", "in", "contains"}
AGGS = {"count", "sum", "avg", "min", "max", "list", "top"}
MAX_EVIDENCE = 400

SYSTEM = (
    "You translate a question about a building drawing into ONE query over the tables you are given. "
    "You never calculate or guess values yourself: the query is executed on exact data extracted from the drawing. "
    "Reply with a single JSON object and nothing else."
)

FORMAT = """QUERY FORMAT
{"table": <table>, "where": [{"col": <column>, "op": "=" | "!=" | ">" | ">=" | "<" | "<=" | "in" | "contains", "val": <value>}],
 "agg": "count" | "sum" | "avg" | "min" | "max" | "list" | "top", "col": <column for sum/avg/min/max/top; for list, a column or a list of columns to show>,
 "group": <optional column, or list of columns: one result per distinct value>, "n": <optional int>, "order": "desc" | "asc"}
- count = number of matching rows; list = show the matching rows; top = the n rows with the largest (desc) or smallest (asc) col.
- Text comparisons ignore case. Use the listed values exactly as written: the drawing may be in German, French,
  Italian, Finnish... so map the question's words onto them (bedroom -> Zimmer / Chambre / Stanza, bathroom -> Bad / Bain / Bagno).
- "where" may be empty. Use only tables and columns that exist.
- If the question asks for several separate facts, reply {"queries": [{"title": "<short label>", <query fields>}, ...]} (at most 5).
- If the tables cannot answer the question, reply {"unanswerable": "<short reason>"}."""

EXAMPLES = {
    "swiss": """EXAMPLES
Q: Which room is the largest? -> {"table":"rooms","where":[{"col":"outdoor","op":"=","val":false}],"agg":"top","col":"area_m2","n":1,"order":"desc"}
Q: How many rooms are bigger than 15 m2? -> {"table":"rooms","where":[{"col":"area_m2","op":">","val":15}],"agg":"count"}
Q: Total living area of each apartment? -> {"table":"apartments","where":[],"agg":"list"}
Q: How many bathrooms have a bathtub? -> {"table":"rooms","where":[{"col":"bathtubs","op":">","val":0}],"agg":"count"}
Q: Which rooms have a shower? -> {"table":"rooms","where":[{"col":"showers","op":">","val":0}],"agg":"list"}
Q: What is the widest door? -> {"table":"doors","where":[],"agg":"top","col":"width_m","n":1,"order":"desc"}
Q: How many rooms open onto a Korridor? (only when a connections table is given) -> {"table":"connections","where":[{"col":"other_label","op":"=","val":"Korridor"},{"col":"kind","op":"=","val":"door"}],"agg":"count"}""",
    "ifc": """EXAMPLES
Q: Which room is the largest? -> {"table":"spaces","where":[],"agg":"top","col":"area_m2","n":1,"order":"desc"}
Q: How many windows per storey? -> {"table":"windows","where":[],"agg":"count","group":"storey"}
Q: What is the narrowest door width? -> {"table":"doors","where":[],"agg":"min","col":"width_m"}
Q: Total length of the exterior walls? -> {"table":"walls","where":[{"col":"external","op":"=","val":true}],"agg":"sum","col":"length_m"}
Q: How many beams are there? -> {"table":"elements","where":[{"col":"ifc_class","op":"=","val":"IfcBeam"}],"agg":"count"}
Q: Which software created this model? -> {"table":"model","where":[{"col":"key","op":"in","val":["authoring_application","ifc_schema"]}],"agg":"list"}
Q: What are the heights of each level? -> {"table":"storeys","where":[],"agg":"list"}
Q: Which rooms are on the first floor, with their areas? -> {"table":"spaces","where":[{"col":"storey","op":"=","val":"<storey name as listed>"}],"agg":"list","col":["name","area_m2"]}
Q: Door types and their counts per storey? -> {"table":"doors","where":[],"agg":"count","group":["type","storey"]}
Q: Which walls are load-bearing? (only when a properties table is given) -> {"table":"properties","where":[{"col":"ifc_class","op":"=","val":"IfcWall"},{"col":"prop","op":"=","val":"LoadBearing"},{"col":"value","op":"=","val":true}],"agg":"list"}
Q: Total volume of the slabs? (properties table) -> {"table":"properties","where":[{"col":"ifc_class","op":"=","val":"IfcSlab"},{"col":"prop","op":"=","val":"NetVolume"}],"agg":"sum","col":"value"}
Q: Materials of the exterior walls? (properties table) -> {"table":"properties","where":[{"col":"ifc_class","op":"=","val":"IfcWall"},{"col":"prop","op":"=","val":"Material"}],"agg":"count","group":"value"}
Q: What is the ceiling height? (properties table; room heights are on IfcSpace) -> {"table":"properties","where":[{"col":"ifc_class","op":"=","val":"IfcSpace"},{"col":"prop","op":"in","val":["FinishCeilingHeight","ClearHeight","Height"]}],"agg":"list"}
- ifc_class filters include subclasses: IfcFlowTerminal also finds IfcSanitaryTerminal, IfcFurnishingElement finds IfcFurniture.
- The properties table must be filtered on "prop" (or "pset"). If the property the question asks about (fire rating, cost,
  U-value...) is not among the properties listed for that table, reply {"unanswerable": "the model has no <property> data"}.""",
}


class PlanError(ValueError):
    pass


def tables_for(path: str) -> tuple[dict, str] | None:
    src = source_of(path)
    if src == "swiss_dwellings":
        return swiss.facts(Path(path).stem), "swiss"
    if path.lower().endswith((".ifc",)) or src in ("ifc_models", "ifc_bench"):
        f = ifc.ifc_path_for(path)
        if f.exists():
            net = mep.network(str(f))  # MEP components and systems, when the model has distribution elements
            tables = {**ifc.facts(str(f)), **(mep.tables(net) if net else {})}
            tables["properties"] = ifc_props.table(str(f))  # property sets and materials (load-bearing, fire rating, volumes...)
            return tables, "ifc"
    return None


def columns(table: dict) -> list[str]:
    cols: dict[str, None] = {}
    for r in table["rows"][:50]:
        for k in list(r) + list(r.get("_lazy", {})):
            if not k.startswith("_"):
                cols[k] = None
    return list(cols)


def cell(row: dict, col: str):
    if col in row:
        return row[col]
    lazy = row.get("_lazy", {})
    if col in lazy:
        row[col] = lazy[col]()
        return row[col]
    return None


def describe_tables(tables: dict) -> str:
    out = []
    for name, t in tables.items():
        rows = t["rows"]
        out.append(f"{name} ({len(rows)} rows) - {t['doc']}")
        for col in columns(t):
            if any(col in r.get("_lazy", {}) for r in rows[:1]):
                continue  # lazy numeric columns: do not force evaluation just to describe them
            vals = [r.get(col) for r in rows]
            if not vals or not all(isinstance(v, (str, bool, type(None))) for v in vals):
                continue
            c = collections.Counter("null" if v is None else str(v) for v in vals)
            if len(c) <= 1 and name != "model":
                continue
            shown = ", ".join(f"{v} x{n}" if n > 1 else v for v, n in c.most_common(24))
            more = f", ... ({len(c)} distinct)" if len(c) > 24 else ""
            out.append(f"    {col}: {shown}{more}")
    return "\n".join(out)


def make_plan(question: str, tables: dict, kind: str) -> tuple[dict, llm.Usage]:
    user = f"TABLES\n{describe_tables(tables)}\n\n{FORMAT}\n\n{EXAMPLES[kind]}\n\nQUESTION: {question}"
    return llm.chat_json(SYSTEM, user, max_tokens=500)


def _match(v, op: str, val) -> bool:
    if op == "in":
        return any(_match(v, "=", x) for x in (val if isinstance(val, list) else [val]))
    if isinstance(val, str) and val.lower() in ("true", "false") and isinstance(v, bool):
        val = val.lower() == "true"
    if isinstance(val, str) and val.lower() in ("null", "none"):
        val = None
    if op in ("=", "!="):
        if isinstance(v, str) and isinstance(val, str):
            eq = v.casefold().strip() == val.casefold().strip()
        elif isinstance(v, bool) or isinstance(val, bool) or v is None or val is None:
            eq = v is val or v == val
        else:
            try:
                eq = float(v) == float(val)
            except (TypeError, ValueError):
                eq = str(v).casefold() == str(val).casefold()
        return eq if op == "=" else not eq
    if op == "contains":
        return v is not None and str(val).casefold() in str(v).casefold()
    try:
        a, b = float(v), float(val)
    except (TypeError, ValueError):
        return False
    return {">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b}[op]


def _property_lines(rows: list[dict]) -> list[str]:
    """Property rows as readable lines: one line per element and value, naming every property that gave that value
    (Height / FinishCeilingHeight / ClearHeight are often the same number), led by a one-line summary when all agree."""
    by_elem: dict = {}
    for r in rows:
        ref = r["_ev"].ref if r.get("_ev") is not None else id(r)
        v = r.get("value")
        key = (ref, json.dumps(v, ensure_ascii=False, default=str))
        if key not in by_elem:
            by_elem[key] = {"label": r.get("type") or r.get("name") or r.get("ifc_class"), "value": v, "unit": r.get("unit"), "props": []}
        if r["prop"] not in by_elem[key]["props"]:
            by_elem[key]["props"].append(r["prop"])

    def val(v, unit=None):
        return _num(v, unit) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)

    lines = [f"{e['label']} — {val(e['value'], e['unit'])} ({' / '.join(e['props'])})" for e in by_elem.values()]
    elems = {k[0] for k in by_elem}
    values = {k[1] for k in by_elem}
    if len(elems) > 1 and len(values) == 1:
        e = next(iter(by_elem.values()))
        props = sorted({p for x in by_elem.values() for p in x["props"]})
        rooms = all(str(x["label"]).startswith("room ") for x in by_elem.values())
        lines.insert(0, f"{val(e['value'], e['unit'])} in all {len(elems)} {'rooms' if rooms else 'elements'} ({' / '.join(props)})")
    return lines


def _where_ok(row: dict, w: dict) -> bool:
    if w["col"] == "ifc_class":  # class filters follow the IFC inheritance tree
        hit = ifc_props.class_match(cell(row, "ifc_class"), w["op"], w.get("val"))
        if hit is not None:
            return hit
    return _match(cell(row, w["col"]), w["op"], w.get("val"))


def _extra(row: dict, table: dict, show: list[str] | None) -> str:
    """The columns a list query asked for that the row text does not already print."""
    if not show:
        return ""
    base = _row_text(row, table)
    out = []
    for c in show:
        v = cell(row, c)
        if v is None:
            continue
        txt = _num(v, table["units"].get(c)) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)
        if txt not in base:
            out.append(f"{c}: {txt}")
    return (" — " + " · ".join(out)) if out else ""


def plan_text(p: dict) -> str:
    where = " and ".join(f"{w['col']} {w['op']} {json.dumps(w['val'], ensure_ascii=False)}" for w in p.get("where") or [])
    agg, col = p.get("agg"), p.get("col")
    tail = {"count": "count", "list": "list", "top": f"top {p.get('n') or 1} by {col} ({p.get('order') or 'desc'})"}.get(agg, f"{agg}({col})")
    group = f" per {p['group']}" if p.get("group") else ""
    return f"{p.get('table')}" + (f" where {where}" if where else "") + f" → {tail}{group}"


def _num(x, unit: str | None) -> str:
    u = {"m2": " m²", "m3": " m³", "m": " m"}.get(unit or "", "")
    return (f"{x:.2f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)) + u


def _row_text(row: dict, table: dict, col: str | None = None) -> str:
    if table.get("show"):
        for c in table["units"]:
            cell(row, c)  # resolve lazy numeric columns the formatter may print
        text = table["show"](row)
        if col and col not in table["units"] and cell(row, col) is not None:
            text += f" — {col}: {cell(row, col)}"
        return text
    label = row["_ev"].label if row.get("_ev") is not None else ""
    if not label:
        label = " ".join(str(v) for k, v in row.items() if not k.startswith("_") and isinstance(v, str))[:60]
    parts = [label]
    shown = [col] if col else [c for c in table["units"] if c in row or c in row.get("_lazy", {})][:2]
    if "value" in row and not col:
        return f"{row.get('key', label)}: {row['value']}"
    for c in shown:
        v = cell(row, c)
        if v is not None:
            parts.append(_num(v, table["units"].get(c)))
    return " — ".join(p for p in parts if p)


def _aggregate(rows: list[dict], agg: str, col: str | None):
    if agg == "count":
        return len(rows), rows
    vals = [(cell(r, col), r) for r in rows]
    vals = [(float(v), r) for v, r in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not vals:
        return None, []
    if agg == "sum":
        return round(sum(v for v, _ in vals), 2), [r for _, r in vals]
    if agg == "avg":
        return round(sum(v for v, _ in vals) / len(vals), 2), [r for _, r in vals]
    best = (min if agg == "min" else max)(v for v, _ in vals)
    return round(best, 3), [r for v, r in vals if v == best]


def execute(p: dict, tables: dict) -> Answer:
    if "unanswerable" in p:
        return Answer(None, None, "unanswerable", 0.0, f"Not answerable from this drawing's data: {p['unanswerable']}")
    name, agg, col = p.get("table"), p.get("agg"), p.get("col")
    if name not in tables:
        raise PlanError(f"unknown table {name!r}")
    if agg not in AGGS:
        raise PlanError(f"unknown aggregation {agg!r}")
    t = tables[name]
    cols = columns(t)
    for w in p.get("where") or []:
        if w.get("col") not in cols or w.get("op") not in OPS:
            raise PlanError(f"bad filter {w}")
    show = None
    if isinstance(col, list):  # several columns asked for: list shows them all; an aggregate uses the first numeric one
        show = [c for c in col if isinstance(c, str) and c in cols]
        bad = [c for c in col if c not in show]
        if bad:
            raise PlanError(f"unknown column(s) {bad}")
        col = next((c for c in show if c in t["units"] or c == "value"), show[0] if show else None) if agg in ("sum", "avg", "min", "max", "top") else None
    if agg in ("sum", "avg", "min", "max", "top") and col not in cols:
        raise PlanError(f"aggregation needs a valid column, got {col!r}")
    group = p.get("group")
    groups = [g for g in (group if isinstance(group, list) else [group]) if g]
    bad = [g for g in groups if g not in cols]
    if bad:
        raise PlanError(f"unknown group column(s) {bad}")

    if name == "properties" and not any(w.get("col") in ("prop", "pset") for w in p.get("where") or []) and "prop" not in groups:
        # without a property filter this lists every property of every element: noise, not an answer
        raise PlanError("the properties table must be filtered on prop (the exact property the question asks about); "
                        "if the model has no such property, the question is unanswerable")
    rows = [r for r in t["rows"] if all(_where_ok(r, w) for w in p.get("where") or [])]
    unit = "count" if agg == "count" else t["units"].get(col or "")
    query = plan_text(p)
    if not rows and agg != "count":
        blank = [w["col"] for w in p.get("where") or [] if all(cell(r, w["col"]) is None for r in t["rows"])]
        why = f"'{blank[0]}' is not recorded in this drawing" if blank else "no rows match"
        return Answer(None, unit, f"plan_{agg}", 0.0, f"{query} · {why}")

    if groups:
        buckets: dict = collections.defaultdict(list)
        for r in rows:
            k = tuple(cell(r, g) for g in groups)
            buckets[k[0] if len(k) == 1 else (None if all(v is None for v in k) else " · ".join("—" if v is None else str(v) for v in k))].append(r)
        lines, used = [], []
        for key in sorted(buckets, key=lambda k: (k is None, str(k))):
            if agg in ("list", "top"):
                v, sub = len(buckets[key]), buckets[key]
            else:
                v, sub = _aggregate(buckets[key], agg, col)
            used += sub
            lines.append(f"{'—' if key is None else key}: {_num(v, None if agg == 'count' else unit) if v is not None else 'n/a'}")
        return Answer("\n".join(lines) or "no matching rows", None, "plan_group", 0.8, query, [r["_ev"] for r in used[:MAX_EVIDENCE] if r.get("_ev")])

    if agg in ("list", "top"):
        if agg == "top":
            ranked = [(cell(r, col), r) for r in rows]
            ranked = [(v, r) for v, r in ranked if isinstance(v, (int, float)) and not isinstance(v, bool)]
            ranked.sort(key=lambda x: x[0], reverse=(p.get("order") or "desc") != "asc")
            rows = [r for _, r in ranked[: int(p.get("n") or 1)]]
        else:
            rows = rows[: int(p.get("n") or 2000)]  # collapsed into one line per distinct text below
        if name == "properties" and agg == "list":
            lines = _property_lines(rows)
        else:
            lines = [_row_text(r, t, col if agg == "top" else None) + _extra(r, t, show) for r in rows]
        if agg == "list":  # identical lines (one element type repeated) become one line with a count
            seen = collections.Counter(lines)
            lines = [f"{ln} ×{seen[ln]}" if seen[ln] > 1 else ln for ln in dict.fromkeys(lines)]
            if len(lines) > 80:
                lines = lines[:80] + [f"… and {len(lines) - 80} more"]
        text = "\n".join(lines) or "no matching rows"
        return Answer(text, None, f"plan_{agg}", 0.8, query, [r["_ev"] for r in rows[:MAX_EVIDENCE] if r.get("_ev")])

    value, used = _aggregate(rows, agg, col)
    if value is None:
        return Answer(None, unit, f"plan_{agg}", 0.0, f"{query} (no numeric values found)")
    if agg in ("min", "max") and used:
        query += " · " + "; ".join(_row_text(r, t) for r in used[:3])
    return Answer(value, unit, f"plan_{agg}", 0.8, query, [r["_ev"] for r in used[:MAX_EVIDENCE] if r.get("_ev")])


def execute_many(queries: list[dict], tables: dict) -> Answer:
    """Multi-part question: run each query, report them as labelled lines."""
    lines, methods, evidence = [], [], []
    for q in queries[:5]:
        try:
            a = execute(q, tables)
        except (PlanError, KeyError, TypeError, ValueError) as e:
            lines.append(f"{q.get('title') or 'query'}: could not be computed ({e})")
            continue
        title = str(q.get("title") or plan_text(q))
        if a.value is None:
            lines.append(f"{title}: not available")
        elif isinstance(a.value, str) and "\n" in a.value:
            lines.append(f"{title}:\n" + "\n".join("  " + ln for ln in a.value.splitlines()))
        else:
            lines.append(f"{title}: {fmt_value(a.value, a.unit)}")
        methods.append(a.method)
        evidence += a.evidence
    if not methods:
        return Answer(None, None, "plan_multi", 0.0, "none of the planned queries could be computed")
    return Answer("\n".join(lines), None, "plan_multi", 0.8, " ; ".join(methods), evidence[:MAX_EVIDENCE])


def answer(path: str, question: str) -> tuple[Answer, llm.Usage | None]:
    got = tables_for(path)
    if got is None:
        return Answer(None, None, "unknown", 0.0, "no structured data for this drawing"), None
    tables, kind = got
    if graph.intent_of(question) is not None:  # topology wording the graph engine could not settle: let the plan use the edges
        g = graph.for_path(path)
        if g is not None:
            tables = {**tables, **graph.tables(g)}
    try:
        p, usage = make_plan(question, tables, kind)
    except llm.LLMError as e:
        return Answer(None, None, "error", 0.0, f"planner unavailable: {e}"), None
    a, err = _run(p, tables)
    problem = err or _empty_reason(a)
    if problem:  # one repair round: show the model what went wrong and the real columns, keep the better result
        try:
            p2, u2 = repair_plan(question, tables, kind, p, problem)
            usage = llm.Usage(usage.model, usage.tokens_in + u2.tokens_in, usage.tokens_out + u2.tokens_out,
                              usage.cost_usd + u2.cost_usd, usage.seconds + u2.seconds)
            a2, err2 = _run(p2, tables)
            if a2 is not None and (a is None or (a.value is None and (a2.value is not None or a2.intent == "unanswerable"))):
                a, err, p = a2, err2, p2
                a.method = f"{a.method} (query repaired after: {problem[:120]})"
        except llm.LLMError:
            pass
    if a is None:
        return Answer(None, None, "error", 0.0, f"planner produced an invalid query ({err}): {json.dumps(p, ensure_ascii=False)[:200]}"), usage
    a.source_file = path
    return a, usage


def _run(p: dict, tables: dict) -> tuple[Answer | None, str | None]:
    try:
        return (execute_many(p["queries"], tables) if isinstance(p.get("queries"), list) and p["queries"] else execute(p, tables)), None
    except (PlanError, KeyError, TypeError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"


def _empty_reason(a: Answer | None) -> str | None:
    """Why an executed query gave nothing worth showing (None when it gave an answer or a reasoned 'unanswerable')."""
    if a is None or a.value is not None or a.intent == "unanswerable":
        return None
    return a.method or "the query returned nothing"


def repair_plan(question: str, tables: dict, kind: str, previous: dict, problem: str) -> tuple[dict, llm.Usage]:
    cols = "\n".join(f"  {name}: {', '.join(columns(t))}" for name, t in tables.items())
    user = (f"TABLES\n{describe_tables(tables)}\n\nCOLUMNS OF EACH TABLE\n{cols}\n\n{FORMAT}\n\n{EXAMPLES[kind]}\n\nQUESTION: {question}\n\n"
            f"YOUR PREVIOUS QUERY:\n{json.dumps(previous, ensure_ascii=False)[:800]}\nIT FAILED: {problem}\n"
            "Reply with a corrected query in the same format: only existing tables and columns, values as listed. "
            "Reply {\"unanswerable\": \"<reason>\"} only if no table can answer it.")
    return llm.chat_json(SYSTEM, user, max_tokens=500)
