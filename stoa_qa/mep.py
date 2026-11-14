"""MEP / services engineer: distribution networks in IFC models (pipes, ducts, fittings, valves, terminals, systems).

Reads what the model states, never estimates:
  components  every IfcDistributionElement: class, type name, systems it belongs to, storey, length, ports
  systems     IfcSystem / IfcDistributionSystem groups and their members (IfcRelAssignsToGroup)
  network     ports (IfcDistributionPort) joined by IfcRelConnectsPorts; a port belongs to its element through
              IfcRelNests (IFC4) or IfcRelConnectsPortToElement (IFC2X3). Components are nodes, port links are edges.

Deterministic answers for the common questions (component breakdown, pipe/duct lengths, systems, network
connectivity, "what is connected to X"); everything else goes to the planner through tables().
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .core import Answer, Evidence, source_of

MAX_EVIDENCE = 400

# friendly names for the IFC classes, IFC4 and IFC2X3
FRIENDLY = {
    "IfcPipeSegment": "pipe segment", "IfcPipeFitting": "pipe fitting", "IfcDuctSegment": "duct segment",
    "IfcDuctFitting": "duct fitting", "IfcValve": "valve", "IfcPump": "pump", "IfcFan": "fan", "IfcTank": "tank",
    "IfcSanitaryTerminal": "sanitary terminal", "IfcWasteTerminal": "waste terminal", "IfcAirTerminal": "air terminal",
    "IfcSpaceHeater": "space heater / radiator", "IfcLightFixture": "light fixture", "IfcOutlet": "outlet",
    "IfcSwitchingDevice": "switching device", "IfcElectricDistributionBoard": "distribution board",
    "IfcCableSegment": "cable segment", "IfcCableCarrierSegment": "cable carrier segment", "IfcSensor": "sensor",
    "IfcFlowMeter": "flow meter", "IfcFilter": "filter", "IfcUnitaryEquipment": "unitary equipment", "IfcBoiler": "boiler",
    "IfcChiller": "chiller", "IfcHeatExchanger": "heat exchanger", "IfcDamper": "damper", "IfcAirTerminalBox": "air terminal box",
    "IfcFlowSegment": "flow segment (pipe/duct)", "IfcFlowFitting": "flow fitting", "IfcFlowTerminal": "flow terminal",
    "IfcFlowController": "flow controller (valve/damper)", "IfcFlowMovingDevice": "flow moving device (pump/fan)",
    "IfcEnergyConversionDevice": "energy conversion device", "IfcFlowStorageDevice": "flow storage device",
    "IfcFlowTreatmentDevice": "flow treatment device", "IfcDistributionControlElement": "control element",
    "IfcDistributionChamberElement": "distribution chamber",
}
SEGMENTS = ("IfcPipeSegment", "IfcDuctSegment", "IfcCableSegment", "IfcCableCarrierSegment", "IfcFlowSegment")


@dataclass
class Component:
    el: object
    cls: str
    type_name: str
    systems: list[str] = field(default_factory=list)
    storey: str | None = None
    length_m: float | None = None
    ports: int = 0
    connected: int = 0

    @property
    def kind(self) -> str:
        return FRIENDLY.get(self.cls, self.cls[3:] if self.cls.startswith("Ifc") else self.cls)

    @property
    def medium(self) -> str:
        """pipe | duct | cable | other, from the class or (IFC2X3 FlowSegment) the type name."""
        t = f"{self.cls} {self.type_name}".lower()
        if "duct" in t or "kanal" in t or "luft" in t:
            return "duct"
        if "pipe" in t or "rohr" in t:
            return "pipe"
        if "cable" in t or "kabel" in t:
            return "cable"
        return "other"

    def ev(self, label: str | None = None, value=None) -> Evidence:
        return Evidence("ifc", self.el.GlobalId, label or f"{self.kind}: {self.type_name}", value, None, self.el.GlobalId)


@dataclass
class Network:
    components: list[Component]
    systems: dict[str, dict]                 # name -> {"type": str, "members": [Component]}
    links: dict[int, set[int]]               # element id -> connected element ids
    ports_total: int = 0
    ports_connected: int = 0
    length_scale: float = 1.0

    def by_id(self) -> dict[int, Component]:
        return {c.el.id(): c for c in self.components}


def _type_name(el) -> str:
    import ifcopenshell.util.element as ue

    t = ue.get_type(el)
    raw = (t.Name if t is not None and t.Name else None) or getattr(el, "ObjectType", None) or el.Name or el.is_a()
    return re.sub(r"\s*:\s*\d+\s*$", "", raw.strip()) or el.is_a()  # Revit appends the element id: "Pipe Types:Cold Water:583067"


def _length(el, scale: float) -> float | None:
    """Segment length from its quantity set or Revit/ArchiCAD property sets, in metres."""
    import ifcopenshell.util.element as ue

    for qs in ue.get_psets(el, qtos_only=True).values():
        for k in ("Length", "NetLength", "GrossLength"):
            v = qs.get(k)
            if isinstance(v, (int, float)) and v > 0:
                return float(v) * scale
    for ps in ue.get_psets(el, psets_only=True).values():
        for k in ("Length", "Länge", "Longueur", "Lunghezza", "Lengte"):
            v = ps.get(k)
            if isinstance(v, (int, float)) and v > 0:
                return float(v) * scale
    return None


def _port_owner(f) -> dict[int, int]:
    owner: dict[int, int] = {}
    for rel in f.by_type("IfcRelNests"):
        host = rel.RelatingObject
        for o in rel.RelatedObjects or ():
            if o.is_a("IfcDistributionPort") or o.is_a("IfcPort"):
                owner[o.id()] = host.id()
    try:
        for rel in f.by_type("IfcRelConnectsPortToElement"):
            owner[rel.RelatingPort.id()] = rel.RelatedElement.id()
    except RuntimeError:
        pass
    return owner


@lru_cache(maxsize=8)
def network(ifc_file: str) -> Network | None:
    from . import ifc

    m = ifc.load_model(ifc_file)
    f = m.f
    els = f.by_type("IfcDistributionElement")
    if not els:
        return None
    comps = {}
    for el in els:
        st = m.storey_of(el)
        c = Component(el, el.is_a(), _type_name(el), [], st.Name if st is not None else None)
        if el.is_a() in SEGMENTS:
            c.length_m = _length(el, m.length_scale)
        comps[el.id()] = c
    systems: dict[str, dict] = {}
    for rel in f.by_type("IfcRelAssignsToGroup"):
        g = rel.RelatingGroup
        if g is None or not g.is_a("IfcSystem") or g.is_a("IfcZone") or g.is_a("IfcStructuralAnalysisModel"):
            continue
        name = (g.Name or g.GlobalId).strip()
        stype = getattr(g, "PredefinedType", None) or getattr(g, "ObjectType", None) or ""
        if stype in ("NOTDEFINED", "USERDEFINED") and getattr(g, "ObjectType", None):
            stype = g.ObjectType
        s = systems.setdefault(name, {"type": str(stype or ""), "members": []})
        for o in rel.RelatedObjects or ():
            c = comps.get(o.id())
            if c is not None:
                s["members"].append(c)
                if name not in c.systems:
                    c.systems.append(name)
    owner = _port_owner(f)
    ports = [p for p in f.by_type("IfcDistributionPort")]
    for p in ports:
        c = comps.get(owner.get(p.id(), -1))
        if c is not None:
            c.ports += 1
    links: dict[int, set[int]] = collections.defaultdict(set)
    connected_ports: set[int] = set()
    for rel in f.by_type("IfcRelConnectsPorts"):
        a, b = rel.RelatingPort, rel.RelatedPort
        if a is None or b is None:
            continue
        connected_ports.update((a.id(), b.id()))
        ea, eb = owner.get(a.id()), owner.get(b.id())
        if ea and eb and ea != eb:
            links[ea].add(eb)
            links[eb].add(ea)
    for pid in connected_ports:
        c = comps.get(owner.get(pid, -1))
        if c is not None:
            c.connected += 1
    return Network(list(comps.values()), systems, dict(links), len(ports), len(connected_ports), m.length_scale)


def for_path(path: str) -> Network | None:
    if source_of(path) not in ("ifc_models", "ifc_bench"):
        return None
    from . import ifc

    f = ifc.ifc_path_for(path)
    return network(str(f)) if f.exists() else None


# ------------------------------------------------------------------ intents (no LLM)

MEP_WORDS = re.compile(
    r"\bmep\b|plumbing|sanitär|ventilat|lüftung|hvac|heating|heizung|duct|pipe|piping|rohr|valve|ventil\b|pump|pumpe|"
    r"radiator|fitting|terminal|sanitary|electrical|distribution|systems?\b|ports?\b|network|connectivity|"
    r"upstream|downstream|feeds?|served by|serves|supply air|return air|exhaust|tanks?\b|filters?\b|heaters?\b|sensors?\b|"
    r"fixtures?\b|(?:flow|water|gas|heat) meters?\b|drains?\b|toilets?\b|washbasins?|urinals?|diffusers?|dampers?|boilers?", re.I)
# properties the model may carry but this agent does not read: leave them to the planner
NOT_MINE = re.compile(r"\bloads?\b|capacity|capacities|\bkw\b|power|voltage|flow rate|pressure|maintenance|insulation|"
                      r"cost|price|efficien|material|diameter|\bsizes?\b|warranty|manufacturer", re.I)
PLUMBING_TYPES = r"toilet|wc\b|closet|urinal|sink|basin|lavator|lavabo|waschtisch|shower|dusche|tub|wanne|bidet|tap|faucet|drain|heater|boiler|plumbing|sanit"
# component words -> IFC classes (longest phrase wins). Generic IFC2X3 classes (IfcFlowTerminal...) only count when
# their type name matches the pattern, since a "flow terminal" can be a light, an air diffuser or a toilet.
CLASS_WORDS = [
    ("pipe fitting", ("IfcPipeFitting", "IfcFlowFitting"), r"pipe|rohr|elbow|tee|bogen"),
    ("duct fitting", ("IfcDuctFitting", "IfcFlowFitting"), r"duct|kanal"),
    ("sanitary terminal", ("IfcSanitaryTerminal",), None), ("waste terminal", ("IfcWasteTerminal",), None),
    ("air terminal", ("IfcAirTerminal", "IfcFlowTerminal"), r"air|diffuser|grille|neck|face|luft"),
    ("unitary equipment", ("IfcUnitaryEquipment",), None),
    ("water heater", ("IfcUnitaryEquipment", "IfcEnergyConversionDevice", "IfcFlowTerminal", "IfcTank"), r"heater|boiler|warmwasser|durchlauf|vitocal"),
    ("boiler", ("IfcBoiler", "IfcEnergyConversionDevice"), r"boiler|kessel"),
    ("light fixture", ("IfcLightFixture", "IfcFlowTerminal"), r"light|lamp|leuchte|lighting"),
    ("lighting", ("IfcLightFixture", "IfcFlowTerminal"), r"light|lamp|leuchte"),
    ("plumbing fixture", ("IfcSanitaryTerminal", "IfcWasteTerminal", "IfcUnitaryEquipment", "IfcFlowTerminal"), PLUMBING_TYPES),
    ("radiator", ("IfcSpaceHeater", "IfcFlowTerminal", "IfcEnergyConversionDevice"), r"radiator|heizkörper|convector"),
    ("space heater", ("IfcSpaceHeater",), None), ("heizkörper", ("IfcSpaceHeater",), None),
    ("fitting", ("IfcPipeFitting", "IfcDuctFitting", "IfcFlowFitting"), None),
    ("pipe", ("IfcPipeSegment", "IfcFlowSegment"), r"pipe|rohr"), ("duct", ("IfcDuctSegment", "IfcFlowSegment"), r"duct|kanal"),
    ("valve", ("IfcValve", "IfcFlowController"), r"valve|ventil|hahn|klappe"), ("damper", ("IfcDamper", "IfcFlowController"), r"damper"),
    ("pump", ("IfcPump", "IfcFlowMovingDevice"), r"pump"), ("fan", ("IfcFan", "IfcFlowMovingDevice"), r"fan|lüfter"),
    ("tank", ("IfcTank", "IfcFlowStorageDevice"), None),
    ("filter", ("IfcFilter", "IfcFlowTreatmentDevice"), None), ("sensor", ("IfcSensor",), None), ("meter", ("IfcFlowMeter",), None),
    ("drain", ("IfcWasteTerminal",), None), ("toilet", ("IfcSanitaryTerminal", "IfcFlowTerminal"), r"toilet|wc\b|closet"),
    ("washbasin", ("IfcSanitaryTerminal", "IfcFlowTerminal"), r"basin|waschtisch|lavabo|lavator|sink"),
    ("terminal", ("IfcSanitaryTerminal", "IfcWasteTerminal", "IfcAirTerminal", "IfcSpaceHeater", "IfcFlowTerminal"), None),
]
GENERIC = ("IfcFlow", "IfcEnergyConversionDevice", "IfcUnitaryEquipment", "IfcDistributionControlElement")
WATER_SUPPLY = r"cold|hot|domestic|supply|pwc|pwh|trinkwasser|kaltwasser|warmwasser|potable"
INTENTS = [
    ("connected_to", r"(?:connected|attached|linked) to|what (?:feeds|supplies|serves)|upstream|downstream|served by"),
    ("connectivity", r"connectivity|connected|connections?\b|unconnected|disconnected|ports?\b|network integration|open ends?"),
    ("length", r"length|how long|länge|longueur"),
    ("systems", r"which (?:mep )?systems|what (?:mep )?systems|systems? (?:are|is) (?:present|installed|in)|system names|count of systems|systems by"),
    ("breakdown", r"component|types?|how many|count|breakdown|comprise|inventory|which .* (?:are|is) (?:in|used)|what .* (?:are|is) (?:in|used)|list"),
]
# one room or space named in the question: this agent has no room geometry, so the planner should take it
ROOM_SCOPED = re.compile(r"\b(?:in|of|inside) (?:the )?(?:room|space|raum)\b|\broom [A-Z]?\d|\bin the (?:bath|kitchen|office)|bathrooms?\b|kitchens?\b", re.I)


def intent_of(question: str) -> str | None:
    if not MEP_WORDS.search(question) or NOT_MINE.search(question) or ROOM_SCOPED.search(question):
        return None
    q = question.lower()
    return next((k for k, pat in INTENTS if re.search(pat, q)), None)


def _scope(net: Network, question: str) -> tuple[list[Component], str] | None:
    """The components the question is about: a named component class first, else a named system kind.
    None when the question names something this model does not have as a distribution element."""
    q = question.lower()
    comps = net.components
    supply = re.search(r"water supply|supply pip|domestic water|drinking water|potable|trinkwasser", q)

    def narrow(hit: list[Component]) -> list[Component]:
        if supply:  # supply pipes only: cold / hot domestic water, not waste or heating
            hit = [c for c in hit if re.search(WATER_SUPPLY, f"{c.type_name} {' '.join(c.systems)}", re.I)]
        return hit

    for word, classes, type_pat in CLASS_WORDS:
        if re.search(r"\b" + re.escape(word) + r"(?:s|es|ing)?\b", q):
            hit = [c for c in comps if c.cls in classes and (not c.cls.startswith(GENERIC) or type_pat is None
                                                              or re.search(type_pat, f"{c.type_name} {c.el.Name or ''}", re.I))]
            if "light" in word:  # switches and panels sit on lighting circuits but are not luminaires
                hit = [c for c in hit if not re.search(r"switch|pole|panel|receptacle|outlet|schalter", f"{c.type_name} {c.el.Name or ''}", re.I)]
            hit = narrow(hit)
            return (hit, f"{word}s") if hit else None
    groups = [
        ("plumbing", r"plumbing|sanit|water|wasser|drain|waste|abwasser", r"pwc|pwh|wasser|water|sanit|abwasser|drain|plumb|regen"),
        ("heating", r"heating|heizung|hot water heating", r"heat|heiz|^h_|vorlauf|rücklauf|wp_"),
        ("ventilation", r"ventilat|lüftung|\bair\b|hvac", r"air|luft|vent|zuluft|abluft|supply|return|exhaust|^l_"),
        ("electrical", r"electric|elektr|cable|outlet|receptacle", r"elec|power|light|kabel|strom"),
    ]
    for name, qpat, spat in groups:
        if re.search(qpat, q):
            in_sys = [c for c in comps if any(re.search(spat, s, re.I) for s in c.systems)]
            if narrow(in_sys):
                return narrow(in_sys), f"{name} systems"
            by_kind = narrow([c for c in comps if re.search(qpat + "|" + spat, f"{c.cls} {c.type_name} {c.kind}", re.I)])
            return (by_kind, f"{name} components") if by_kind else None
    return comps, "all distribution elements"


def _per_storey(comps: list[Component]) -> str:
    c = collections.Counter(x.storey or "no storey" for x in comps)
    return "; ".join(f"{k}: {v}" for k, v in sorted(c.items()))


def answer(path: str, question: str) -> Answer | None:
    intent = intent_of(question)
    if intent is None:
        return None
    net = for_path(path)
    if net is None:
        if intent in ("systems", "length") or re.search(r"\bmep\b|pip|duct|system|plumbing|heating|ventilat|cooling", question, re.I):
            return Answer("None in this model: it has no MEP distribution elements (pipes, ducts, fittings, terminals) and no systems.",
                          None, f"mep_{intent}", 0.85, "Searched the model for IfcDistributionElement and IfcSystem: none found.", [], path)
        return None  # e.g. toilets in an architectural model: rooms may still answer it (planner)
    scoped = _scope(net, question)
    if scoped is None:  # names something that is not a distribution element here (proxies, furniture): planner's turn
        return None
    comps, scope = scoped
    by_level = bool(re.search(r"level|storey|floor|geschoss|ebene|étage", question, re.I))

    if intent == "connectivity":
        ports = sum(c.ports for c in comps)
        conn = sum(c.connected for c in comps)
        loose = [c for c in comps if c.ports and c.connected < c.ports]
        if not net.ports_total:
            return Answer(None, None, "mep_connectivity", 0.6, "The model defines no distribution ports, so connectivity cannot be read from it.", source_file=path)
        rate = conn / ports if ports else 0
        text = (f"{ports:,} ports, {conn:,} connected ({rate:.1%})\n{len(comps):,} components, {ports / max(len(comps), 1):.1f} ports per component\n"
                f"{len(loose):,} components with at least one open port")
        return Answer(text, None, "mep_connectivity", 0.9,
                      f"Ports of {scope} (IfcDistributionPort) and their IfcRelConnectsPorts links; open ports are listed as evidence.",
                      [c.ev(f"open port: {c.kind}", f"{c.connected}/{c.ports}") for c in loose[:MAX_EVIDENCE]], path)

    if intent == "length":
        segs = [c for c in comps if c.cls in SEGMENTS]
        with_len = [c for c in segs if c.length_m]
        if not with_len:
            return Answer(None, None, "mep_length", 0.4, f"No lengths are stored on the {len(segs)} segments of {scope}.", source_file=path)
        groups: dict[str, list[Component]] = collections.defaultdict(list)
        for c in with_len:
            key = c.storey or "no storey" if by_level else f"{c.medium}: {c.type_name}"
            groups[key].append(c)
        total = sum(c.length_m for c in with_len)
        lines = [f"Total: {total:,.1f} m in {len(with_len)} segments"] + [
            f"  {k}: {sum(c.length_m for c in v):,.1f} m ({len(v)} segments)" for k, v in sorted(groups.items(), key=lambda kv: -sum(c.length_m for c in kv[1]))]
        miss = len(segs) - len(with_len)
        return Answer("\n".join(lines), None, "mep_length", 0.9 if not miss else 0.7,
                      f"Sum of the Length stored on each segment of {scope}, converted from the model's unit (×{net.length_scale:g} to m)"
                      + (f"; {miss} segments have no stored length." if miss else "."),
                      [c.ev(value=round(c.length_m, 2)) for c in with_len[:MAX_EVIDENCE]], path)

    if intent == "systems":
        rows = []
        for name, s in sorted(net.systems.items()):
            mem = [c for c in s["members"] if c in comps] if scope != "all distribution elements" else s["members"]
            if mem or scope == "all distribution elements":
                rows.append((name, s["type"], mem))
        if not rows:
            kinds = collections.Counter(c.kind for c in comps)
            return Answer("No IfcSystem groups are defined; components present: " + ", ".join(f"{k} ({v})" for k, v in kinds.most_common()),
                          None, "mep_systems", 0.7, "The model assigns no components to systems; listed the distribution elements instead.",
                          [c.ev() for c in comps[:MAX_EVIDENCE]], path)
        by_type = collections.Counter(t or "untyped" for _, t, _ in rows)
        text = f"{len(rows)} systems" + (" — " + ", ".join(f"{t}: {n}" for t, n in by_type.most_common()) if len(by_type) > 1 else "") + "\n"
        text += "\n".join(f"{n}" + (f" ({t})" if t else "") + f" — {len(m)} components" for n, t, m in rows[:60])
        return Answer(text, None, "mep_systems", 0.9, f"IfcSystem groups and their members (IfcRelAssignsToGroup), {scope}.",
                      [c.ev() for _, _, m in rows for c in m][:MAX_EVIDENCE], path)

    if intent == "connected_to":
        from . import ifc as ifc_mod

        target = _named_component(net, question)
        if target is None:
            return Answer(None, None, "mep_connected_to", 0.0, "Name the component (its IFC name or type) to trace from.", source_file=path)
        ids = net.by_id()
        near = [ids[i] for i in net.links.get(target.el.id(), ()) if i in ids]
        text = f"{target.kind} '{target.el.Name or target.type_name}' connects to {len(near)} component(s)\n" + "\n".join(
            f"{c.kind}: {c.el.Name or c.type_name}" for c in near)
        return Answer(text, None, "mep_connected_to", 0.85, "Direct port-to-port links (IfcRelConnectsPorts).",
                      [target.ev("traced from")] + [c.ev() for c in near], path)

    # breakdown: components by class (and type), optionally per storey
    by_cls = collections.Counter(c.kind for c in comps)
    lines = [f"{len(comps):,} components ({scope})"]
    for kind, n in by_cls.most_common():
        types = collections.Counter(c.type_name for c in comps if c.kind == kind)
        detail = "; ".join(f"{t} ×{k}" for t, k in types.most_common(4)) + (f"; +{len(types) - 4} more types" if len(types) > 4 else "")
        lines.append(f"  {kind}: {n} — {detail}")
    if by_level:
        lines.append("By storey: " + _per_storey(comps))
    plain_count = (re.search(r"^how many\b", question.strip(), re.I) and len(by_cls) == 1
                   and not re.search(r"breakdown|types?\b|each|\bper\b|\bby\b|distribut", question, re.I))
    value = len(comps) if plain_count else "\n".join(lines)
    return Answer(value, "count" if isinstance(value, int) else None, "mep_breakdown", 0.9,
                  f"IfcDistributionElement instances of {scope}, grouped by IFC class and type.",
                  [c.ev() for c in comps[:MAX_EVIDENCE]], path)


def _named_component(net: Network, question: str) -> Component | None:
    q = question.casefold()
    best = None
    for c in net.components:
        for name in (c.el.Name, c.type_name, c.el.Tag if hasattr(c.el, "Tag") else None):
            if name and len(name) > 3 and name.casefold() in q:
                if best is None or len(name) > best[0]:
                    best = (len(name), c)
    return best[1] if best else None


# ------------------------------------------------------------------ for the planner

def tables(net: Network) -> dict:
    rows = [{"ifc_class": c.cls, "kind": c.kind, "type_name": c.type_name, "system": ", ".join(c.systems) or None,
             "storey": c.storey, "length_m": round(c.length_m, 3) if c.length_m else None, "medium": c.medium,
             "ports": c.ports, "connected_ports": c.connected, "_ev": c.ev()} for c in net.components]
    sys_rows = [{"system": n, "system_type": s["type"] or None, "components": len(s["members"]),
                 "_ev": Evidence("ifc", "", n, len(s["members"]))} for n, s in net.systems.items()]
    return {
        "mep_components": {"rows": rows, "units": {"length_m": "m"},
                           "doc": "MEP distribution elements: ifc_class, kind (pipe segment, valve, sanitary terminal...), type_name, "
                                  "system (names of the systems it belongs to), storey, length_m (segments), medium (pipe|duct|cable|other), "
                                  "ports, connected_ports"},
        "mep_systems": {"rows": sys_rows, "units": {}, "doc": "MEP systems (IfcSystem): system name, system_type, components (member count)"},
    }
