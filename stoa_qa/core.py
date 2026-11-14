"""Shared types: every solver returns an Answer with evidence a human can check."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

DATASET = Path(os.environ.get(
    "STOA_DATASET",
    Path(__file__).resolve().parent.parent / "junction-dataset/full/junction-dataset",
))


@dataclass
class Evidence:
    """One thing we counted or measured, located on the source drawing.

    kind:   "polygon" (sheet coords), "ifc" (GlobalId), "text" (label anchor)
    ref:    stable id linking 2D <-> 3D (DXF handle, IFC GlobalId, ...)
    """
    kind: str
    ref: str
    label: str = ""
    value: float | None = None
    coords: list[list[float]] | None = None


@dataclass
class Answer:
    value: float | int | str | None
    unit: str | None = None
    intent: str = ""
    confidence: float = 0.0
    method: str = ""          # human-readable "how we computed this"
    evidence: list[Evidence] = field(default_factory=list)
    source_file: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


UNANSWERED = Answer(value=None, intent="unknown", confidence=0.0, method="no solver matched")


def source_of(path: str) -> str:
    """'data/swiss_dwellings/x.pdf' -> 'swiss_dwellings'."""
    return Path(path).parts[1]
