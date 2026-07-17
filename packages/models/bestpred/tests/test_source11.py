from __future__ import annotations

import pytest

from bestpred.core.source11 import simulate_source11_records
from bestpred.io.parameters import read_parameters
from bestpred.io.source11 import read_source11_examples


@pytest.mark.source11
def test_read_source11_examples(source11_fixture_dir):
    examples = read_source11_examples(source11_fixture_dir / "DCRexample.txt")

    assert len(examples) == 43
    assert examples[0].number == 1
    assert examples[0].plan_lines[0].name == "255-day RIP"
    assert examples[1].plan_lines[0].name == "LER sampled"
    assert examples[1].plan_lines[1].name == "odd months"


@pytest.mark.source11
def test_simulate_first_source11_record_matches_fortran_segment(source11_fixture_dir):
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    examples = read_source11_examples(source11_fixture_dir / "DCRexample.txt")

    records = simulate_source11_records(examples, parameters)
    first = records[0]

    assert first.cow_id == "HOUSA.EX.COW.0001"
    assert first.birth_date == "19930401"
    assert first.fresh_date == "19990401"
    assert first.parity == 5
    assert first.length == 255
    assert first.previous_days_open == 140
    assert len(first.segments) == 9
    assert first.segments[0].to_fortran_segment() == " 1510222 1100 779343046"
    assert first.segments[-1].to_fortran_segment() == "25510222 1100 691363235"


@pytest.mark.source11
def test_simulate_multiline_source11_record(source11_fixture_dir):
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    examples = read_source11_examples(source11_fixture_dir / "DCRexample.txt")

    records = simulate_source11_records(examples, parameters)
    second = records[1]

    assert second.parity == 4
    assert second.length == 300
    assert len(second.segments) == 10
    assert {segment.times_sampled for segment in second.segments} == {0, 2}
