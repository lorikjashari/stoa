"""Spatial graph of a drawing: which rooms connect, through which door, and which only share a wall.

Nodes are rooms (plus 'outside' for doors that lead off the drawing); edges say how two rooms relate:

    door    a door leaf sits between them (via = the door)
    open    they touch with no wall between them (open-plan kitchen / living room)
    wall    they share a wall longer than MIN_WALL metres, without a door

Questions about topology ("Which rooms open onto the corridor?", "How do I get from 2.13 to the stairwell?",
"Which rooms have no window?") are answered by traversing this graph, never by an LLM looking at pixels.
Every answer cites the rooms and doors it used, so the route can be checked on the sheet.

Sources:
  swiss_dwellings  DXF geometry from stoa_qa.swiss (the shared reader; this module only reads it)
  ifc_models       IfcRelSpaceBoundary (door / opening / virtual boundaries); door geometry where those are missing
"""
from __future__ import annotations

import heapq
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .core import Answer, Evidence, source_of

DOOR_REACH = 0.3   # m: a door touches the rooms within this distance of its leaf
DOOR_SECOND = 0.3  # the second room must overlap the door at least this fraction as much as the first
OPEN_GAP = 0.03    # m: rooms closer than this have no wall between them
WALL_GAP = 0.45    # m: rooms closer than this share a wall (Swiss walls are 0.1-0.4 m thick)
MIN_WALL = 0.5     # m of shared wall before two rooms count as neighbours
MAX_LIST = 60
HOW = {"door": "door", "stair": "stairs", "open": "open, no wall"}
STAIR = re.compile(r"stair|treppe|escalier|scala|trap|schody|scara|trepp|trapp", re.I)


@dataclass
class Node:
    id: str
    kind: str                     # room | outside
    label: str                    # printed label ("Küche"), or the IFC name
    number: str | None = None     # printed room number ("2.14"), IFC LongName/Name
    group: str | None = None      # apartment letter (Swiss) or storey (IFC)
    area: float | None = None
    windows: int = 0
    center: tuple[float, float] | None = None
    ev: Evidence | None = None

    @property
    def name(self) -> str:
        if self.kind == "outside":
            return self.label
        num = f"{self.number} " if self.number and self.number != self.label else ""
        return f"{num}{self.label}".strip()


@dataclass
class Edge:
    a: str
    b: str
    kind: str                     # door | open | wall | stair (IFC: a stair space joining two storeys)
    via: Evidence | None = None   # the door, for kind == door
    length: float | None = None   # shared wall length (m), for kind == wall / open
    center: tuple[float, float] | None = None  # where the connection is (door centre)


@dataclass
class Graph:
    source: str
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    geometric_links: int = 0      # IFC: door-room links found from geometry instead of space boundaries

    def rooms(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.kind == "room"]

    def links(self, nid: str, kinds=("door", "open", "stair")) -> list[tuple[str, Edge]]:
        out = []
        for e in self.edges:
            if e.kind in kinds and nid in (e.a, e.b):
                out.append((e.b if e.a == nid else e.a, e))
        return out


# ------------------------------------------------------------------ builders

def _swiss(stem: str) -> Graph:
    from shapely.geometry import Point

    from . import swiss

    fl = swiss.load_floor(stem)
    g = Graph("swiss_dwellings")
    apt_of_unit = {h: next((l for l, p in fl.apt_tags if u.contains(Point(p))), None) for h, u in fl.units}
    rooms = fl.numbered
    for r in rooms:
        apt = next((apt_of_unit[h] for h, u in fl.units if r.anchor and u.contains(Point(r.anchor))), None)
        c = r.poly.representative_point()
        g.nodes[f"room:{r.number}"] = Node(
            f"room:{r.number}", "room", r.label or "(unlabelled)", r.number, apt, round(r.area, 2), 0, (c.x, c.y),
            Evidence("polygon", r.handle, f"{r.number} {r.label or ''}".strip(), round(r.area, 2), swiss._coords(r.poly), f"room:{r.number}"))
    outside = Node("outside", "outside", "outside this floor plan")
    door_pairs: set[tuple[str, str]] = set()
    for i, (h, p) in enumerate(fl.doors):
        reach = p.buffer(DOOR_REACH)
        hits = sorted(((reach.intersection(r.poly).area, r) for r in rooms if reach.intersects(r.poly)), key=lambda x: -x[0])
        hits = [r for a, r in hits if a >= DOOR_SECOND * hits[0][0]][:2] if hits else []
        c = p.centroid
        ev = Evidence("polygon", h, f"door {i + 1}", round(swiss._long_side(p), 2), swiss._coords(p), f"door:{h}")
        if not hits:
            continue
        a = f"room:{hits[0].number}"
        if len(hits) == 1:
            g.nodes.setdefault("outside", outside)
            b = "outside"
        else:
            b = f"room:{hits[1].number}"
        pair = tuple(sorted((a, b)))
        if any(tuple(sorted((e.a, e.b))) == pair and e.center and math.dist(e.center, (c.x, c.y)) < 1.2 for e in g.edges):
            continue  # second leaf of a double door, or the same door drawn twice
        g.edges.append(Edge(a, b, "door", ev, None, (c.x, c.y)))
        door_pairs.add(pair)
    for i, r in enumerate(rooms):
        grown = r.poly.buffer(WALL_GAP)
        for s in rooms[i + 1:]:
            if not grown.intersects(s.poly):
                continue
            d = r.poly.distance(s.poly)
            shared = min(s.poly.exterior.intersection(r.poly.buffer(WALL_GAP)).length,
                         r.poly.exterior.intersection(s.poly.buffer(WALL_GAP)).length)
            pair = tuple(sorted((f"room:{r.number}", f"room:{s.number}")))
            if d < OPEN_GAP and shared >= MIN_WALL / 2 and pair not in door_pairs:
                g.edges.append(Edge(*pair, "open", None, round(shared, 2)))
            elif shared >= MIN_WALL and pair not in door_pairs:
                g.edges.append(Edge(*pair, "wall", None, round(shared, 2)))
    for h, p in fl.windows:
        reach = p.buffer(DOOR_REACH)
        best = max(rooms, key=lambda r: reach.intersection(r.poly).area, default=None)
        if best is not None and reach.intersects(best.poly):
            g.nodes[f"room:{best.number}"].windows += 1
    return g


def _footprint(m, el):
    """2D footprint (world XY) and z-range of an IFC element, from its triangulated geometry."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    geo = m.shape(el)
    if geo is None or not geo.verts:
        return None
    v, fc = geo.verts, geo.faces
    pts = [(v[i], v[i + 1], v[i + 2]) for i in range(0, len(v), 3)]
    tris = []
    for i in range(0, len(fc), 3):
        t = Polygon([pts[fc[i]][:2], pts[fc[i + 1]][:2], pts[fc[i + 2]][:2]])
        if t.area > 1e-6:
            tris.append(t)
    if not tris:
        return None
    zs = [p[2] for p in pts]
    return unary_union(tris).buffer(0), min(zs), max(zs)


def _ifc(ifc_file: str) -> Graph | None:
    """Rooms = IfcSpace. Doors join the spaces whose boundaries name them (IfcRelSpaceBoundary); a door bounded
    by fewer than two spaces, or a model without boundaries, falls back to geometry: the spaces whose footprint
    lies within DOOR_REACH + half a wall of the door, on the door's level."""
    from . import ifc

    m = ifc.load_model(ifc_file)
    f = m.f
    spaces = f.by_type("IfcSpace")
    doors = [d for d in m.products(["IfcDoor"])]
    if not spaces or not doors:
        return None
    g = Graph("ifc")
    for sp in spaces:
        name = (sp.LongName or sp.Name or sp.GlobalId).strip()
        num = (sp.Name or "").strip() or None
        st = m.storey_of(sp)
        try:
            area = m.space_area(sp)[0]
        except Exception:
            area = None
        g.nodes[sp.GlobalId] = Node(sp.GlobalId, "room", name, num, st.Name if st is not None else None,
                                    round(area, 2) if area else None, 0, None,
                                    Evidence("ifc", sp.GlobalId, name, round(area, 2) if area else None, None, sp.GlobalId))
    fills = {rel.RelatingOpeningElement.id(): rel.RelatedBuildingElement for rel in f.by_type("IfcRelFillsElement")}
    by_door: dict[int, set[str]] = {}
    virtual: dict[int, set[str]] = {}
    windows: dict[str, set[int]] = {}
    for bd in f.by_type("IfcRelSpaceBoundary"):
        sp, el = bd.RelatingSpace, bd.RelatedBuildingElement
        if sp is None or not sp.is_a("IfcSpace") or sp.GlobalId not in g.nodes:
            continue
        if el is None or getattr(bd, "PhysicalOrVirtualBoundary", None) == "VIRTUAL" or el.is_a("IfcVirtualElement"):
            if el is not None:
                virtual.setdefault(el.id(), set()).add(sp.GlobalId)
            continue
        if el.is_a("IfcOpeningElement") and el.id() in fills:
            el = fills[el.id()]
        if el.is_a("IfcDoor"):
            by_door.setdefault(el.id(), set()).add(sp.GlobalId)
        elif el.is_a("IfcWindow"):
            windows.setdefault(sp.GlobalId, set()).add(el.id())
    for sid, ws in windows.items():
        g.nodes[sid].windows = len(ws)

    feet: dict[str, tuple] = {}

    def foot(sp):
        if sp.GlobalId not in feet:
            feet[sp.GlobalId] = _footprint(m, sp)
        return feet[sp.GlobalId]

    geometric = 0
    for door in doors:
        known = set(by_door.get(door.id(), ()))
        df = _footprint(m, door)  # also gives the door's position, used for walking distances
        center = (df[0].centroid.x, df[0].centroid.y) if df is not None and not df[0].is_empty else None
        if len(known) < 2:
            if df is not None:
                dpoly, z0, z1 = df
                near = []
                for sp in spaces:
                    fp = foot(sp)
                    if fp is None or not (fp[1] - 0.3 <= z0 <= fp[2]):
                        continue
                    d = fp[0].distance(dpoly)
                    if d <= DOOR_REACH:
                        near.append((d, -fp[0].intersection(dpoly.buffer(DOOR_REACH)).area, sp.GlobalId))
                near.sort()
                for _, _, gid in near:
                    if len(known) >= 2:
                        break
                    if gid not in known:
                        known.add(gid)
                        geometric += 1
        if not known:
            continue
        ev = Evidence("ifc", door.GlobalId, door.Name or "door", None, None, door.GlobalId)
        s = sorted(known)
        if len(s) == 1:
            g.nodes.setdefault("outside", Node("outside", "outside", "outside (no room on the other side)"))
            g.edges.append(Edge(s[0], "outside", "door", ev, None, center))
        for i, a in enumerate(s):
            for b in s[i + 1:]:
                g.edges.append(Edge(a, b, "door", ev, None, center))
    # rooms that touch with no wall between them, and stairs that join storeys: from the space footprints
    from shapely.strtree import STRtree

    for sp in spaces:
        foot(sp)
    have = [(sp.GlobalId, fp) for sp in spaces if (fp := feet.get(sp.GlobalId)) is not None]
    for gid, fp in have:
        c = fp[0].representative_point()
        g.nodes[gid].center = (c.x, c.y)
    if have:
        tree = STRtree([fp[0] for _, fp in have])
        linked = {tuple(sorted((e.a, e.b))) for e in g.edges}
        for i, (a, fa) in enumerate(have):
            for j in tree.query(fa[0].buffer(WALL_GAP)):
                j = int(j)
                if j <= i:
                    continue
                b, fb = have[j]
                pair = tuple(sorted((a, b)))
                if pair in linked:
                    continue
                z_overlap = min(fa[2], fb[2]) - max(fa[1], fb[1])
                overlap = fa[0].intersection(fb[0]).area
                if z_overlap > 0.5 and overlap > 0.05 * min(fa[0].area, fb[0].area):
                    continue  # one space drawn over another (zones, gross-area spaces): not a neighbour
                gap = fa[0].distance(fb[0]) if z_overlap > 0.5 else None
                if gap is not None and gap < OPEN_GAP * 2:
                    shared = fa[0].boundary.intersection(fb[0].buffer(OPEN_GAP * 2)).length
                    if shared >= MIN_WALL:
                        g.edges.append(Edge(a, b, "open", None, round(shared, 2)))
                        linked.add(pair)
                elif gap is not None and gap < WALL_GAP:
                    shared = min(fa[0].boundary.intersection(fb[0].buffer(WALL_GAP)).length,
                                 fb[0].boundary.intersection(fa[0].buffer(WALL_GAP)).length)
                    if shared >= MIN_WALL:
                        g.edges.append(Edge(a, b, "wall", None, round(shared, 2)))
                elif z_overlap <= 0.5 and (STAIR.search(g.nodes[a].label) or STAIR.search(g.nodes[b].label)):
                    if overlap >= 1.0 and abs(fa[1] - fb[1]) < 6.0:
                        g.edges.append(Edge(a, b, "stair"))
                        linked.add(pair)
    for spaces_v in virtual.values():
        s = sorted(spaces_v)
        for i, a in enumerate(s):
            for b in s[i + 1:]:
                g.edges.append(Edge(a, b, "open"))
    g.geometric_links = geometric
    return g


@lru_cache(maxsize=64)
def for_path(path: str) -> Graph | None:
    """The graph for a dataset drawing, or None when its format has no topology we can read."""
    src = source_of(path)
    if src == "swiss_dwellings":
        return _swiss(Path(path).stem)
    if src in ("ifc_models", "ifc_bench"):
        from . import ifc

        f = ifc.ifc_path_for(path)
        return _ifc(str(f)) if f.exists() else None
    return None


# ------------------------------------------------------------------ queries

def neighbours(g: Graph, nid: str, kinds=("door", "open", "stair")) -> list[tuple[Node, Edge]]:
    seen, out = set(), []
    for other, e in g.links(nid, kinds):
        if (other, e.kind) not in seen:
            seen.add((other, e.kind))
            out.append((g.nodes[other], e))
    return out


def _dist(a: tuple | None, b: tuple | None) -> float:
    return math.dist(a, b) if a and b else 1.0


def walk_points(g: Graph, path: list[tuple[str, Edge | None]]) -> list[tuple[float, float]] | None:
    """The walked line: start room centre -> each door in turn -> goal room centre (or the last door, when it leads out).

    Straight lines door to door, the way a person crosses a corridor; open connections and stairs have no
    door position, so the route then passes through that room's centre."""
    pts = [g.nodes[path[0][0]].center]
    for n, e in path[1:]:
        pts.append(e.center if e is not None and e.center else g.nodes[n].center)
    last = g.nodes[path[-1][0]]
    if last.kind == "room":
        pts.append(last.center)
    if len(pts) < 2 or any(p is None for p in pts):
        return None
    return [tuple(p) for p in pts]


def walk_length(g: Graph, path: list[tuple[str, Edge | None]]) -> float | None:
    """Metres along walk_points."""
    pts = walk_points(g, path)
    return sum(math.dist(p, q) for p, q in zip(pts, pts[1:])) if pts else None


def route(g: Graph, start: str, goals: set[str]) -> list[tuple[str, Edge | None]] | None:
    """Shortest walk from start to any goal through doors and open connections: [(node, edge used to reach it)]."""
    best = {start: 0.0}
    prev: dict[str, tuple[str, Edge]] = {}
    heap = [(0.0, start)]
    while heap:
        d, n = heapq.heappop(heap)
        if n in goals:
            path = [(n, prev[n][1] if n in prev else None)]
            while n in prev:
                n = prev[n][0]
                path.append((n, prev[n][1] if n in prev else None))
            return path[::-1]
        if d > best.get(n, math.inf):
            continue
        for other, e in g.links(n):
            if other == "outside" and other not in goals:
                continue  # never route through the outside of the drawing
            here = e.center or g.nodes[n].center
            step = _dist(g.nodes[n].center, here) + _dist(here, g.nodes[other].center) + 0.01
            if d + step < best.get(other, math.inf):
                best[other] = d + step
                prev[other] = (n, e)
                heapq.heappush(heap, (d + step, other))
    return None


def entrances(g: Graph) -> list[tuple[Edge, Node, Node]]:
    """Doors that lead out: (door, inside room, outside room).

    Swiss: doors that leave an apartment (to the stairwell, a common corridor, a cellar) or the drawing.
    IFC: doors with no space on the other side, i.e. to the outside of the building."""
    out = []
    for e in g.edges:
        if e.kind != "door":
            continue
        a, b = g.nodes[e.a], g.nodes[e.b]
        if b.kind == "outside":
            out.append((e, a, b))
        elif a.kind == "outside":
            out.append((e, b, a))
        elif g.source != "swiss_dwellings":
            continue
        elif a.group and not b.group:
            out.append((e, a, b))
        elif b.group and not a.group:
            out.append((e, b, a))
        elif a.group and b.group and a.group != b.group:
            out.append((e, a, b))
    return out


# ------------------------------------------------------------------ question understanding (no LLM)

# generic room words -> labels as printed in the dataset (DE/FR/IT/EN, the Swiss and IFC vocabularies)
ROOM_WORDS = {
    "corridor": ["korridor", "gang", "flur", "diele", "couloir", "corridoio", "corridor", "hall", "hallway", "entrée", "ingresso", "vorplatz"],
    "kitchen": ["küche", "cuisine", "cucina", "kitchen"],
    "living": ["wohnen", "wohnen/essen", "wohnzimmer", "séjour", "salon", "soggiorno", "living", "living room"],
    "bedroom": ["zimmer", "schlafzimmer", "chambre", "camera", "stanza", "bedroom"],
    "bathroom": ["bad", "badezimmer", "dusche", "salle de bain", "salle de bains", "bagno", "bathroom", "wc", "toilet"],
    "stair": ["treppenhaus", "treppe", "cage d'escalier", "escalier", "vano scala", "scala", "stair", "stairs", "staircase", "stairwell"],
    "storage": ["reduit", "abstellraum", "réduit", "ripostiglio", "storage", "keller", "cave", "cantina"],
    "balcony": ["balkon", "balcon", "balcone", "loggia", "terrasse", "terrazza", "terrace", "balcony"],
}
QUESTION_WORDS = {
    "corridor": r"corridor|hallway|\bhall\b|korridor|\bflur\b|couloir|corridoio",
    "kitchen": r"kitchen|küche|kuche|cuisine|cucina",
    "living": r"living|wohnen|séjour|salon|soggiorno",
    "bedroom": r"bedroom|zimmer|chambre|camera|stanza",
    "bathroom": r"bathroom|\bbath\b|\bbad\b|salle de bain|bagno|\bwc\b|toilet",
    "stair": r"stair|treppe|escalier|scala",
    "storage": r"storage|reduit|réduit|abstell|ripostiglio|keller",
    "balcony": r"balcon|balkon|loggia|terrace|terrass",
}
EXIT = r"\bexit\b|entrance|front door|way out|ausgang|eingang|sortie|entrée de l'appartement|uscita|ingresso"
INTENTS = [
    ("route", r"\broute\b|\bpath\b|distance (?:from|between|to)|walking distance|nearest (?:emergency )?exit|escape|evacuat|fluchtweg|entfernung|distanza|itin[ée]raire|\bway\b|get (?:from|to)|walk|\bgo from\b|how (?:do|can) i (?:get|reach)|reach|weg|wie komme|chemin|comment aller|percorso|come arrivo|doors? (?:do i|to) (?:pass|go)"),
    ("no_window", r"no windows?|without (?:a )?windows?|windowless|natural light|daylight|fensterlos|ohne fenster|sans fen[eê]tre|senza finestr"),
    ("entrances", r"(?:how many|which|list|where|count|wie viele|combien|quant)[^?]*(?:entrance doors?|entry doors?|front doors?|entrances|eingangst[üu]r|porte d'entr[ée]e|porta d'ingresso"
                  r"|doors? (?:that )?(?:lead|leads|open|opens|go|goes) (?:to the )?(?:outside|exterior)|exterior doors?|outside doors?|external doors?|außent[üu]r)"),
    ("dimension", r"how wide|\bwidth\b|how long is|\blength of (?:the )?(?:room|corridor|hall|kitchen|bedroom|bathroom|[0-9])|dimensions?\b|"
                  r"how big is (?:the )?(?:room|corridor|hall|kitchen|bedroom|bathroom|[0-9])|wie breit|largeur|larghezza"),
    ("adjacent", r"adjacent|next to|neighbou?r|border|share[sd]? a wall|beside|benachbart|angrenz|neben|voisin|attigu|adiacent|confin"),
    ("connects", r"connect|opens? (?:on|in)to|opens? to the|lead(?:s)? (?:in)?to|access(?:ed)? from|accessible from|doors? between|reachable|verbunden|f[üu]hrt (?:in|zu)|reli[ée]|donne sur|collegat|comunicano|through which"),
]


QUOTED = re.compile(r"['\"‘’“”«»„‚][^'\"‘’“”«»„‚]{2,80}['\"‘’“”«»„‚]")


def intent_of(question: str) -> str | None:
    """Topology intent of a question, or None. Quoted names ('Escape Route Test Model') are ignored."""
    q = QUOTED.sub(" ", question).lower()
    if re.search(r"interior doors? and|together|zusammen|innent[üu]r|ensemble|insieme|schedule|type [a-z]?\d", q):
        return None  # counting every door, or a schedule lookup: not a topology question
    return next((k for k, pat in INTENTS if re.search(pat, q)), None)


def _norm(s: str) -> str:
    return " ".join(s.casefold().split())


def find_rooms(g: Graph, text: str) -> list[list[Node]]:
    """Room references in the text, in order of appearance: each is the list of rooms it may mean.

    Room numbers ("2.14"), quoted names, printed labels ("Küche") and generic words ("the kitchen") all count;
    a span already taken by a more specific reference is not read again."""
    t = text.casefold()
    rooms = g.rooms()
    refs: list[tuple[int, list[Node]]] = []
    taken: list[tuple[int, int]] = []

    def add(start: int, end: int, hit: list[Node]) -> None:
        if hit and not any(a < end and start < b for a, b in taken):
            taken.append((start, end))
            refs.append((start, hit))

    numbers = sorted({r.number for r in rooms if r.number and re.search(r"\d", r.number)}, key=len, reverse=True)
    for num in numbers:  # printed room numbers: "2.14" (Swiss), "1E12" / "A102" (IFC space names)
        for m in re.finditer(r"(?<![\w.])" + re.escape(num) + r"(?![\w.])", text, re.I):
            add(m.start(), m.end(), [r for r in rooms if r.number == num])
    for m in re.finditer(r"['\"‘’“”«»„‚]([^'\"‘’“”«»„‚]{2,60})['\"‘’“”«»„‚]", text):
        q = _norm(m.group(1))
        add(m.start(), m.end(), [r for r in rooms if _norm(r.label) == q or _norm(r.name) == q] or [r for r in rooms if q in _norm(r.name)])
    labels = sorted({r.label for r in rooms if len(r.label) > 2 and not r.label.startswith("(")}, key=len, reverse=True)
    for lab in labels:  # printed labels mentioned without quotes ("Küche"), longest first
        for m in re.finditer(r"(?<!\w)" + re.escape(lab.casefold()) + r"(?!\w)", t):
            add(m.start(), m.end(), [r for r in rooms if r.label == lab])
    for kind, pat in QUESTION_WORDS.items():  # generic words: "the kitchen", "the corridor"
        for m in re.finditer(pat, t):
            words = ROOM_WORDS[kind]
            add(m.start(), m.end(), [r for r in rooms if _norm(r.label) in words or any(w in _norm(r.label) for w in words if len(w) > 3)])
    refs.sort(key=lambda x: x[0])
    return [h for _, h in refs]


def _room_list(nodes: list[Node]) -> str:
    return ", ".join(n.name for n in nodes)


def _door_ev(e: Edge | None) -> list[Evidence]:
    return [e.via] if e is not None and e.via is not None else []


def answer(path: str, question: str) -> Answer | None:
    """A topology answer, or None when the question is not about how rooms connect."""
    intent = intent_of(question)
    if intent is None:
        return None
    g = for_path(path)
    if g is None:
        return Answer(None, None, f"graph_{intent}", 0.0, "This drawing has no room topology we can read (no room/door geometry or IFC space boundaries).", source_file=path)
    refs = find_rooms(g, question)
    method_src = ("DXF room outlines and door leaves" if g.source == "swiss_dwellings" else
                  "IFC space boundaries" + (f", plus {g.geometric_links} door links from geometry" if g.geometric_links else ""))

    if intent == "no_window":
        rooms = [r for r in g.rooms() if r.windows == 0 and r.kind == "room"]
        if refs:
            rooms = [r for r in rooms if r in refs[0]] if "which" not in question.lower() else rooms
        text = "\n".join(r.name for r in rooms[:MAX_LIST]) or "every room has a window"
        return Answer(text if "how many" not in question.lower() else len(rooms), None if "how many" not in question.lower() else "count",
                      "graph_no_window", 0.85, f"Rooms with no window touching their outline ({method_src}): {len(rooms)}",
                      [r.ev for r in rooms[:MAX_LIST] if r.ev], path)

    if intent == "dimension":
        if not refs:
            return Answer(None, None, "graph_dimension", 0.0, "Name the room to measure (a room number like 2.14 or its label).", source_file=path)
        from shapely.geometry import Polygon

        lines, ev = [], []
        for r in refs[0][:12]:
            if not (r.ev and r.ev.coords and len(r.ev.coords) >= 4):
                continue
            poly = Polygon(r.ev.coords)
            rect = poly.minimum_rotated_rectangle
            xs = list(rect.exterior.coords)
            a_, b_ = math.dist(xs[0], xs[1]), math.dist(xs[1], xs[2])
            w, l = min(a_, b_), max(a_, b_)
            shape = "" if poly.area >= 0.85 * rect.area else " — not a plain rectangle, so parts of it are narrower"
            lines.append(f"{r.name}: {w:.2f} m wide × {l:.2f} m long ({poly.area:.2f} m²){shape}")
            ev.append(r.ev)
        if not lines:
            return Answer(None, None, "graph_dimension", 0.0, "The room outlines of this model are not in the room graph, so they cannot be measured here.",
                          source_file=path)
        return Answer("\n".join(lines), None, "graph_dimension", 0.85,
                      "Width and length of the smallest rectangle around each room outline (DXF geometry); wall finishes not deducted.", ev, path)

    if intent == "entrances":
        es = entrances(g)
        if refs:  # "entrance doors of apartment A" is handled below; a named room limits it to that room's doors
            want = {n.id for n in refs[0]}
            es = [x for x in es if x[1].id in want or x[2].id in want] or es
        m = re.search(r"(?:apartment|wohnung|appartement|appartamento)\s+([A-Z])\b", question, re.I)
        if m:
            es = [x for x in es if (x[1].group or "").upper() == m.group(1).upper()]
        per: dict[str, list[str]] = {}
        for e, inner, outer in es:
            per.setdefault(inner.group or "-", []).append(f"{inner.name} → {outer.name}")
        what = "apartment" if g.source == "swiss_dwellings" else "storey"
        lines = [f"{what} {k}: " + "; ".join(v) for k, v in sorted(per.items())]
        lead = "Doors that lead out of an apartment or off the drawing" if g.source == "swiss_dwellings" else "Doors with no room on the other side (exterior doors)"
        return Answer(len(es), "count", "graph_entrances", 0.85,
                      f"{lead} ({method_src}). " + " | ".join(lines),
                      [ev for e, *_ in es for ev in _door_ev(e)], path)

    if intent == "route":
        if not refs:
            return Answer(None, None, "graph_route", 0.0, "Name the rooms to route between (a room number like 2.14 or its label).", source_file=path)
        if len(refs) == 1:
            if not re.search(EXIT, question.lower()):
                return Answer(None, None, "graph_route", 0.0, "Only one room recognised; name where to start and where to go.", source_file=path)
            starts = refs[0]
            group = starts[0].group if g.source == "swiss_dwellings" else None  # IFC groups are storeys: exits may be on another one
            goals = {outer.id for _, inner, outer in entrances(g) if group is None or inner.group == group}
        else:
            starts, goals = refs[0], {n.id for n in refs[1]}
        best = None
        for s in starts:
            p = route(g, s.id, goals - {s.id})
            if p and (best is None or (walk_length(g, p) or len(p)) < (walk_length(g, best) or len(best))):
                best = p
        if best is None:
            return Answer(None, None, "graph_route", 0.3, f"No route through doors between those rooms on this drawing ({method_src}).", source_file=path)
        doors = [e for _, e in best if e is not None and e.kind == "door"]
        steps = [g.nodes[n].name for n, _ in best]
        ev = []
        for n, e in best:
            ev += _door_ev(e)
            if g.nodes[n].ev:
                ev.append(g.nodes[n].ev)
        metres = walk_length(g, best)
        line = walk_points(g, best)
        if line:  # drawn as the route on the sheet (pipeline.overlay); not a numbered item
            ev.insert(0, Evidence("route", "route", "walking route", round(metres, 1) if metres else None,
                                  [[round(x, 3), round(y, 3)] for x, y in line]))
        far = f", about {metres:.0f} m walking door to door from the room centre" if metres else ""
        return Answer(" → ".join(steps) + (f"  (≈ {metres:.0f} m)" if metres else ""), None, "graph_route", 0.9,
                      f"Shortest walk through doors and open connections ({method_src}): {len(best) - 1} step(s), {len(doors)} door(s){far}.",
                      ev, path)

    # adjacent / connects
    if not refs:
        return Answer(None, None, f"graph_{intent}", 0.0, "Name the room to look at (a room number like 2.14 or its label).", source_file=path)
    kinds = ("door", "open", "stair", "wall") if intent == "adjacent" else ("door", "open", "stair")
    targets = refs[0]
    if len(refs) >= 2 and intent == "connects":  # "Is there a door between X and Y?"
        other = {n.id for n in refs[1]}
        found = [(t, n, e) for t in targets for n, e in neighbours(g, t.id, kinds) if n.id in other]
        ev = [x.ev for t, n, _ in found for x in (t, n) if x.ev] + [ev for *_, e in found for ev in _door_ev(e)]
        text = "\n".join(f"{t.name} ↔ {n.name} ({HOW.get(e.kind, 'open, no wall')})" for t, n, e in found)
        return Answer("yes\n" + text if found else "no", None, "graph_connects", 0.9,
                      f"Direct connections between the two rooms ({method_src}).", ev, path)
    lines, ev = [], []
    for t in targets[:12]:
        ns = neighbours(g, t.id, kinds)
        if not ns:
            continue
        part = []
        for n, e in ns:
            how = {"door": "door", "open": "open", "stair": "stairs", "wall": f"wall {e.length:.1f} m" if e.length else "wall"}[e.kind]
            part.append(f"{n.name} ({how})")
            if n.ev:
                ev.append(n.ev)
            ev += _door_ev(e)
        lines.append(f"{t.name}: " + ", ".join(part) if len(targets) > 1 else "\n".join(part))
        if t.ev:
            ev.insert(0, t.ev)
    if not lines:
        return Answer("none", None, f"graph_{intent}", 0.8, f"No {'neighbouring' if intent == 'adjacent' else 'connected'} rooms found ({method_src}).", [t.ev for t in targets if t.ev], path)
    what = "rooms sharing a door, an opening or a wall" if intent == "adjacent" else "rooms reached through a door or an opening"
    return Answer("\n".join(lines), None, f"graph_{intent}", 0.9, f"{what.capitalize()} ({method_src}).", ev[:400], path)


# ------------------------------------------------------------------ for the planner and the API

def tables(g: Graph) -> dict:
    rows = []
    for e in g.edges:
        a, b = g.nodes[e.a], g.nodes[e.b]
        ev = e.via or b.ev
        for x, y in ((a, b), (b, a)):
            rows.append({"room": x.name, "room_number": x.number, "room_label": x.label, "other": y.name,
                         "other_number": y.number, "other_label": y.label, "kind": e.kind,
                         "group": x.group, "shared_wall_m": e.length, "_ev": ev})
    return {"connections": {
        "rows": rows, "units": {"shared_wall_m": "m"},
        "show": lambda r: f"{r['room']} ↔ {r['other']} ({r['kind']})",
        "doc": "room-to-room connections, one row per direction: room/room_number/room_label, other/other_number/other_label, "
               "kind (door = a door between them, open = no wall between them, wall = they only share a wall), "
               "group (apartment or storey), shared_wall_m. Use it for adjacency and 'which rooms open onto X'."}}


def to_json(g: Graph) -> dict:
    return {"source": g.source,
            "nodes": [{"id": n.id, "kind": n.kind, "name": n.name, "group": n.group, "area_m2": n.area, "windows": n.windows,
                       "center": n.center} for n in g.nodes.values()],
            "edges": [{"a": e.a, "b": e.b, "kind": e.kind, "door": e.via.viewer_id if e.via else None,
                       "shared_wall_m": e.length, "center": e.center} for e in g.edges]}
