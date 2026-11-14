"""Entry point: answer(path, question) -> Answer, dispatched by drawing source/format."""
from __future__ import annotations

import traceback

from . import cubicasa, ifc, swiss
from .core import UNANSWERED, Answer, source_of

SOLVERS = {
    "swiss_dwellings": swiss.answer,
    "ifc_models": ifc.answer,
    "cubicasa5k": cubicasa.answer,
}


def answer(path: str, question: str) -> Answer:
    solver = SOLVERS.get(source_of(path))
    if solver is None:
        return UNANSWERED
    try:
        return solver(path, question)
    except Exception as e:  # never crash the batch; surface the failure as an unanswered result
        return Answer(None, None, "error", 0.0, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=2)[-300:]}")
