"""Room graph (stoa_qa.graph) on the synthetic floor from conftest:

    1.03 Balkon  (0..2, 3..4)   touches 1.01 along y = 3, no wall: open
    1.01 Zimmer  (0..4, 0..3)   door at x = 4 to 1.02, window on its south side
    1.02 Bad     (4..6, 0..2)   window on its east side
"""
from tests.conftest import STEM

PATH = f"data/swiss_dwellings/{STEM}.pdf"


def _graph():
    from stoa_qa import graph

    return graph.for_path(PATH)


def _kinds(g):
    return {tuple(sorted((g.nodes[e.a].number or "", g.nodes[e.b].number or ""))): e.kind for e in g.edges}


def test_door_and_open_connections_come_from_the_geometry():
    g = _graph()
    links = _kinds(g)
    assert links[("1.01", "1.02")] == "door"
    assert links[("1.01", "1.03")] == "open"
    assert ("1.02", "1.03") not in links


def test_route_goes_through_the_door_and_the_opening():
    from stoa_qa import graph

    a = graph.answer(PATH, "How do I get from room 1.02 to the Balkon?")
    assert a.value.startswith("1.02 Bad → 1.01 Zimmer → 1.03 Balkon"), a.method
    assert a.evidence[0].kind == "route" and len(a.evidence[0].coords) >= 3  # the walked line, drawn on the sheet
    assert any(e.viewer_id and e.viewer_id.startswith("door:") for e in a.evidence)


def test_neighbours_windows_and_door_checks():
    from stoa_qa import graph

    a = graph.answer(PATH, "Which rooms are adjacent to room 1.01?")
    assert "1.02 Bad (door)" in a.value and "1.03 Balkon (open)" in a.value
    assert graph.answer(PATH, "Which rooms have no window?").value == "1.03 Balkon"
    assert graph.answer(PATH, "Is there a door between 1.01 and 1.02?").value.startswith("yes")
    assert graph.answer(PATH, "Wie komme ich von 1.02 zum Balkon?").value.startswith("1.02 Bad → 1.01 Zimmer")


def test_dataset_style_questions_are_not_taken_for_topology():
    from stoa_qa import graph

    for q in ["How many doors (interior doors and entrance doors together) are drawn on this floor plan?",
              "Wie viele Türen (Innentüren und Eingangstüren zusammen) sind auf diesem Grundriss eingezeichnet?",
              "How many doors are there in the model 'Real-World Escape Route Test Model (Vienna)'?",
              "On sheet A-801, how many doors in the door schedule are door type A1 (swing door)?",
              "What is the net floor area of room 1.01 ('Zimmer') in m²?"]:
        assert graph.intent_of(q) is None, q


def test_pipeline_and_api_use_the_graph():
    from fastapi.testclient import TestClient

    from stoa_qa.api import app

    c = TestClient(app)
    r = c.post("/api/ask", json={"id": PATH, "question": "Which rooms open onto room 1.01?"}).json()
    assert r["engine"]["kind"] == "graph" and r["engine"]["llm_tokens"] == 0
    assert r["overlay"]["items"]
    g = c.get("/api/graph", params={"id": PATH}).json()
    assert g["counts"]["door"] == 1 and g["counts"]["open"] == 1
    assert g["space"] == "px" and all(n["px"] for n in g["nodes"])
    r = c.post("/api/ask", json={"id": PATH, "question": "How many doors (interior doors and entrance doors together) are drawn on this floor plan?"}).json()
    assert r["engine"]["kind"] == "geometry" and r["correct"] is True
