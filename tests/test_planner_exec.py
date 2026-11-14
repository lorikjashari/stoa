"""Planner query execution on synthetic tables: column lists, multi-column groups, IFC class inheritance."""
import pytest

from stoa_qa import planner


def _tables():
    rows = [
        {"name": "Kitchen", "storey": "L1", "type": "A", "ifc_class": "IfcSanitaryTerminal", "area_m2": 12.5},
        {"name": "Bath", "storey": "L1", "type": "B", "ifc_class": "IfcFurniture", "area_m2": 4.0},
        {"name": "Bed", "storey": "L2", "type": "A", "ifc_class": "IfcWall", "area_m2": 15.0},
    ]
    return {"t": {"rows": rows, "units": {"area_m2": "m2"}, "doc": "test"}}


def test_list_with_several_columns_shows_them():
    a = planner.execute({"table": "t", "where": [{"col": "storey", "op": "=", "val": "L1"}], "agg": "list", "col": ["name", "area_m2"]}, _tables())
    assert "Kitchen" in a.value and "12.5 m²" in a.value and "Bed" not in a.value


def test_group_by_two_columns():
    a = planner.execute({"table": "t", "where": [], "agg": "count", "group": ["type", "storey"]}, _tables())
    assert a.value.splitlines() == ["A · L1: 1", "A · L2: 1", "B · L1: 1"]


def test_ifc_class_filter_includes_subclasses():
    t = _tables()
    a = planner.execute({"table": "t", "where": [{"col": "ifc_class", "op": "=", "val": "IfcFlowTerminal"}], "agg": "count"}, t)
    b = planner.execute({"table": "t", "where": [{"col": "ifc_class", "op": "=", "val": "IfcFurnishingElement"}], "agg": "count"}, t)
    assert a.value == 1 and b.value == 1


def test_unknown_columns_still_fail_loudly():
    with pytest.raises(planner.PlanError):
        planner.execute({"table": "t", "where": [], "agg": "count", "group": ["type", "colour"]}, _tables())
    with pytest.raises(planner.PlanError):
        planner.execute({"table": "t", "where": [], "agg": "sum", "col": "volume_m3"}, _tables())
