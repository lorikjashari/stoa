"""Language-independent question understanding via multilingual keyword stems.

The dataset asks the same things in ~10 languages. Instead of one regex per
language we detect (a) the object class, (b) the measured quantity and
(c) the scope, then let each solver resolve names against its own index.
"""
from __future__ import annotations

import re

# object class -> stems (lowercase, matched at word start)
OBJECTS: dict[str, list[str]] = {
    "door": ["door", "tür", "deur", "dør", "døre", "puert", "port", "uks", "ust", "drzwi", "uși", "usi", "ovi"],
    "window": ["window", "fenster", "raam", "ramen", "vindue", "ventan", "finestr", "fenêtre", "fenetre", "aken", "akn", "okn", "okien", "okno", "ferestr", "ikkun"],
    "space": ["room", "space", "räum", "raum", "ruimte", "rum", "local", "ruum", "pomieszcze", "stanz", "încăper", "pièce", "piece"],
    "wall": ["wall", "wänd", "wand", "wanden", "væg", "muro", "muri", "mur", "sein", "ścian", "scian", "pereț", "peret"],
    "column": ["column", "stütz", "kolom", "søjl", "pilar", "pilastr", "colonn", "post", "słup", "slup", "stâlp", "stalp"],
    "beam": ["beam", "träger", "trager", "balk", "bjælk", "viga", "trav", "tala", "belk", "belek", "grinz"],
    "stair": ["stair", "trepp", "trap", "escaler", "scal", "schod", "scări", "scari"],
    "slab": ["slab", "decke", "bodenplatt", "vloer", "dæk", "plade", "plader", "losa", "solai", "płyt", "plyt", "placă", "plac"],
    "sanitary": ["sanit"],
    "light": ["light fixture", "lighting", "luminaire", "iluminat", "leucht", "verlicht", "armatur", "lys"],
}

MODEL_WORDS = ["model", "modell", "modellen", "modelo", "modello", "mudel", "modelu", "modelul"]


def stem_hit(text: str, stems: list[str]) -> bool:
    t = text.lower()
    return any(re.search(r"(?<![\wäöüåæøéèàìòùąćęłńóśźżăâîșț])" + re.escape(s), t) for s in stems)


def objects_in(text: str) -> list[str]:
    return [k for k, stems in OBJECTS.items() if stem_hit(text, stems)]


def quantity_of(text: str) -> str:
    t = text.lower()
    if "m³" in t or "m3" in t:
        return "volume"
    if "m²" in t or "m2" in t or "ft²" in t or "sq" in t:
        return "area"
    if re.search(r"\b(metres|meters|metre|meter|metern|metros|metri|metrach|metri)\b", t) or re.search(r"\b(length|länge|lang|længde|lengte|longitud|lunghezza|pikkus|długość|lungime)\b", t):
        return "length"
    return "count"


def is_model_scope(text: str) -> bool:
    t = text.lower()
    return any(re.search(r"\b" + w + r"\b", t) for w in MODEL_WORDS)


QUOTES = re.compile(r"«([^»]+)»|„([^“”]+)[“”]|\"([^\"]+)\"")


def quoted(text: str) -> list[str]:
    """Quoted names in «», „“, „” or "". Single quotes are ambiguous (d'escalier) and are
    handled by solvers via substring matching against known names."""
    return [next(g for g in m.groups() if g) for m in QUOTES.finditer(text)]
