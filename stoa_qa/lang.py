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
    # "scala" (IT), not "scal": that also matched English "scale"/"scaled" ("What is the scale?" -> 1 IfcStair)
    "stair": ["stair", "trepp", "trap", "escaler", "scala", "schod", "scări", "scari"],
    "slab": ["slab", "decke", "bodenplatt", "vloer", "dæk", "plade", "plader", "losa", "solai", "płyt", "plyt", "placă", "plac"],
    "sanitary": ["sanit"],
    "light": ["light fixture", "lighting", "luminaire", "iluminat", "leucht", "verlicht", "armatur", "lys"],
}

MODEL_RE = re.compile(r"\b(?:model|mudel)\w*", re.I)  # model, Modell, modellen, modelo, modello, modelu, modelul, mudelis


def stem_hit(text: str, stems: list[str]) -> bool:
    t = text.lower()
    return any(re.search(r"(?<![\wäöüåæøéèàìòùąćęłńóśźżăâîșț])" + re.escape(s), t) for s in stems)


def objects_in(text: str) -> list[str]:
    return [k for k, stems in OBJECTS.items() if stem_hit(text, stems)]


# "how many …" in the dataset's languages: the question counts, whatever else it mentions
COUNT_OPENER = re.compile(r"how many|number of|wie viele|anzahl|combien|nombre de|quant[ie]|numero di|hoeveel|aantal|hvor mange|antal|"
                          r"kuinka monta|mitu|ile |liczba|cu[áa]nt[oa]s|n[úu]mero de|c[âa]te|c[âa]ți|num[ăa]rul", re.I)
AREA_WORDS = re.compile(r"\bareas?\b|surface|superfici|fl(?:ä|ae)che|oppervlak|\bareal|pindala|powierzchn|suprafa", re.I)
VOLUME_WORDS = re.compile(r"\bvolum|inhoud|rumfang|\bmaht|objęto|objeto", re.I)


def quantity_of(text: str) -> str:
    t = text.lower()
    if "m³" in t or "m3" in t:
        return "volume"
    if "m²" in t or "m2" in t or "ft²" in t or "sq" in t:
        return "area"
    if re.search(r"\b(metres|meters|metre|meter|metern|metros|metri|metrach|metri)\b", t) or re.search(r"\b(length|länge|lang|længde|lengte|longitud|lunghezza|pikkus|długość|lungime)\b", t):
        return "length"
    if COUNT_OPENER.search(t):
        return "count"
    # no unit written: "What is the net floor area of apartment B?" is still an area question
    if AREA_WORDS.search(t):
        return "area"
    if VOLUME_WORDS.search(t):
        return "volume"
    return "count"


def is_model_scope(text: str) -> bool:
    return bool(MODEL_RE.search(text))


QUOTES = re.compile(r"«([^»]+)»|„([^“”]+)[“”]|\"([^\"]+)\"")


def quoted(text: str) -> list[str]:
    """Quoted names in «», „“, „” or "". Single quotes are ambiguous (d'escalier) and are
    handled by solvers via substring matching against known names."""
    return [next(g for g in m.groups() if g) for m in QUOTES.finditer(text)]
