"""What a question is scoped to beyond its drawing: an apartment, or one room.

The deterministic solvers count and sum over a whole floor, storey or model. A question that narrows the scope
("windows in the kitchen", "bathrooms in apartment B") must be answered inside that scope or declined, never
answered with the floor-wide number. (Before this module: 28 windows "in the kitchen", at 0.97 confidence.)

Rooms are found with the room graph's own reader (graph.find_rooms: room numbers, printed labels, quoted names,
generic words in EN/DE/FR/IT), so "the kitchen" means the same rooms here as in topology answers.
"""
from __future__ import annotations

import re

# "apartment B", "Wohnung B", "Whg. B", "appartement B", "appartamento B": the letter is case-sensitive
APARTMENT = re.compile(r"(?i:\b(?:apartments?|flat|wohnung|whg\.?|appartements?|appartamento|apt\.?)\s*)([A-Z])\b(?![.\d])")
ASIDE = re.compile(r"\([^)]*\)")                                   # "(WC)", "(residential units)"
# "… except balconies, loggias and terraces": what follows is excluded, not where to look
EXCEPT = re.compile(r"\b(?:except|excluding|exclusive of|without|apart from|other than|not counting|ohne|au(?:ss|ß)er|ausgenommen|"
                    r"exkl\w*|sauf|hors|à l'exclusion|sans|eccetto|escluso|esclusi|escluse|tranne|senza)\b.*$", re.I | re.S)
QUOTED = re.compile(r"«[^»]*»|„[^“”]*[“”]|\"[^\"]*\"|(?<!\w)'[^']{1,80}'(?!\w)")


# "living area", "Wohnfläche", "surface habitable": a kind of area, not the living room
LIVING = re.compile(r"\bliving (?:area|space|floor area|surface)|\bhabitable (?:area|space|floor area)|\bliveable (?:area|space)|"
                    r"wohnfl(?:ä|ae)che|surface (?:habitable|d'habitation)|superficie (?:abitabile|residenziale|habitable)", re.I)
# other area phrases that name no room ("usable area", "Nutzfläche", "surface utile")
AREA_PHRASE = re.compile(r"\b(?:usable|useful|floor|net|gross|total) (?:floor )?(?:area|space)|nutzfl(?:ä|ae)che|surface utile|"
                         r"superficie utile", re.I)
# "the total area of all kitchens": a sum over every room that matches
TOTAL = re.compile(r"\b(?:all|total|sum|together|combined|alle[ns]?|gesamt\w*|insgesamt|zusammen|tou(?:s|tes)|totale?|"
                   r"ensemble|tutt[ie]|complessiv\w*|somma)\b", re.I)


def living(question: str) -> bool:
    """Does the question ask for the living area (the apartments' floor area)?"""
    return bool(LIVING.search(question))


def wants_total(question: str) -> bool:
    return bool(TOTAL.search(question))


def apartment(question: str) -> str | None:
    """The apartment letter the question is about, or None."""
    m = APARTMENT.search(QUOTED.sub(" ", question))
    return m.group(1) if m else None


def clean(text: str, keep_quotes: bool = True) -> str:
    """Text without asides in parentheses (and without quoted names unless keep_quotes)."""
    text = AREA_PHRASE.sub(" ", LIVING.sub(" ", EXCEPT.sub(" ", ASIDE.sub(" ", text))))
    return text if keep_quotes else QUOTED.sub(" ", text)


def rooms_named(path: str, text: str, group: str | None = None):
    """Room references in `text`, each a list of graph Nodes it may mean (optionally only rooms in `group`,
    the apartment letter on Swiss floors). [] when the drawing has no room graph or names no room."""
    from . import graph

    g = graph.for_path(path)
    if g is None:
        return []
    refs = graph.find_rooms(g, clean(text))
    if group is not None:
        refs = [[n for n in ref if n.group == group] for ref in refs]
    return [ref for ref in refs if ref]


def mentions_room(text: str) -> bool:
    """Cheap test (no drawing needed): does the text name a room by number or a generic room word?"""
    from . import graph

    t = clean(text, keep_quotes=False).lower()
    return bool(re.search(r"(?<![\w.])U?\d+\.\d+(?![\w.])|\broom\s+\S*\d", t)) or \
        any(re.search(p, t) for p in graph.QUESTION_WORDS.values())


def which(nodes) -> str:
    """'2.02 Küche, 2.15 Küche' for a decline message."""
    names = [n.name for n in nodes[:8]]
    return ", ".join(names) + (f" and {len(nodes) - 8} more" if len(nodes) > 8 else "")
