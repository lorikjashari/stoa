"""The crew: MEP engineer (stoa_qa.mep), auditor (stoa_qa.audit) and the trace the pipeline reports."""
import pytest

from tests.conftest import STEM

SWISS = f"data/swiss_dwellings/{STEM}.pdf"


@pytest.fixture(scope="module")
def mep_model(tmp_path_factory):
    """Two cold-water pipes joined through a ball valve: 6 ports, 4 of them connected, lengths 2.5 m + 1.5 m (in mm)."""
    import ifcopenshell.api as api

    f = api.run("project.create_file", version="IFC4")
    proj = api.run("root.create_entity", f, ifc_class="IfcProject", name="t")
    api.run("unit.assign_unit", f)  # millimetres
    bldg = api.run("root.create_entity", f, ifc_class="IfcBuilding", name="b")
    st = api.run("root.create_entity", f, ifc_class="IfcBuildingStorey", name="Level 1")
    api.run("aggregate.assign_object", f, products=[bldg], relating_object=proj)
    api.run("aggregate.assign_object", f, products=[st], relating_object=bldg)
    pipe1, pipe2, valve = [api.run("root.create_entity", f, ifc_class=c, name=n) for c, n in
                           [("IfcPipeSegment", "Pipe:Cold Water:1"), ("IfcPipeSegment", "Pipe:Cold Water:2"), ("IfcValve", "Valve:Ball:3")]]
    api.run("spatial.assign_container", f, products=[pipe1, pipe2, valve], relating_structure=st)
    ports = {e.Name: (api.run("system.add_port", f, element=e), api.run("system.add_port", f, element=e)) for e in (pipe1, pipe2, valve)}
    api.run("system.connect_port", f, port1=ports[pipe1.Name][1], port2=ports[valve.Name][0])
    api.run("system.connect_port", f, port1=ports[valve.Name][1], port2=ports[pipe2.Name][0])
    system = api.run("system.add_system", f, ifc_class="IfcDistributionSystem")
    system.Name = "Domestic cold water"
    api.run("system.assign_system", f, products=[pipe1, pipe2, valve], system=system)
    for pipe, length in ((pipe1, 2500.0), (pipe2, 1500.0)):
        ps = api.run("pset.add_pset", f, product=pipe, name="Dimensions")
        api.run("pset.edit_pset", f, pset=ps, properties={"Length": length})
    path = tmp_path_factory.mktemp("mep") / "plumbing.ifc"
    f.write(str(path))
    return str(path)


@pytest.fixture
def ask_mep(mep_model, monkeypatch):
    from stoa_qa import mep

    monkeypatch.setattr(mep, "for_path", lambda path: mep.network(mep_model))
    return lambda q: mep.answer("data/ifc_bench/plumbing.ifc", q)


def test_mep_breakdown_lengths_systems_and_ports(ask_mep):
    a = ask_mep("Which component types make up the plumbing system and how many of each?")
    assert "pipe segment: 2" in a.value and "valve: 1" in a.value and "Pipe:Cold Water" in a.value
    a = ask_mep("What is the total length of pipe used in the building?")
    assert a.value.startswith("Total: 4.0 m in 2 segments"), a.value  # 2500 + 1500 mm
    a = ask_mep("Which MEP systems are installed?")
    assert "Domestic cold water" in a.value
    a = ask_mep("How well connected is the pipe network?")
    assert "4 ports, 2 connected" in a.value or "6 ports, 4 connected" in a.value, a.value
    assert len(a.evidence) == 2  # the two pipes each have one open end
    a = ask_mep("What is connected to Valve:Ball:3?")
    assert "connects to 2 component(s)" in a.value


def test_mep_leaves_other_questions_to_the_planner():
    from stoa_qa import mep

    assert mep.intent_of("What is the total cost of the valves?") is None          # a property it does not read
    assert mep.intent_of("How many radiators are in room A202?") is None           # one room: needs room geometry
    assert mep.intent_of("How many square meters of flooring are needed?") is None
    assert mep.intent_of("What is the net floor area of the room 'Kitchen'?") is None


def test_auditor_confirms_a_label_count_from_the_printed_labels():
    from stoa_qa import api, pipeline

    r = pipeline.respond(api._drawing(SWISS), "How many rooms labelled 'Zimmer' are there on this floor?")
    checks = {c["name"]: c for c in r["audit"]}
    assert checks["printed labels"]["status"] == "pass" and checks["marked items"]["status"] == "pass"
    assert r["confidence"] >= 0.95
    assert [c["agent"] for c in r["crew"]] == ["Quantity surveyor", "Auditor"]


def test_auditor_fails_an_answer_of_the_wrong_kind():
    from stoa_qa import audit
    from stoa_qa.core import Answer

    bad = Answer(7, "count", "plan_count", 0.8, "a planner slip")
    checks = audit.answer_shape("What is the net floor area of room 1.01 in m²?", bad)
    assert checks and checks[0].status == "fail"
    assert audit.answer_shape("How many doors?", Answer(2.5, "count")) [0].status == "fail"


def test_crew_trace_shows_who_declined():
    from stoa_qa import api, pipeline

    r = pipeline.respond(api._drawing(SWISS), "How do I get from room 1.02 to the Balkon?")
    assert r["crew"][0]["agent"] == "Navigator" and r["crew"][0]["outcome"] == "answered"
    assert r["engine"]["kind"] == "graph"
