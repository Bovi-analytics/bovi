from __future__ import annotations

import math

import pytest

from bestpred.io.dcr import read_dcr_results, write_compatibility_dcr


@pytest.mark.golden
def test_read_current_fortran_dcr_results(source11_fixture_dir):
    rows = read_dcr_results(source11_fixture_dir / "results_v2.dcr")

    assert len(rows) == 43
    assert rows[0].animal_id == "HOUSA.EX.COW."
    assert rows[0].fresh_date == "19990401"
    assert rows[0].dim == 255
    assert rows[0].numeric_values[3] == 21900
    assert rows[0].numeric_values[38] == 3.08


@pytest.mark.golden
def test_write_compatibility_dcr_round_trips_raw_lines(source11_fixture_dir, tmp_path):
    source = source11_fixture_dir / "results_v2.dcr"
    rows = read_dcr_results(source)
    output = tmp_path / "results_v2.dcr"

    write_compatibility_dcr(output, rows)

    assert output.read_text() == source.read_text()


@pytest.mark.legacy
def test_legacy_manual_output_is_not_current_oracle(source11_fixture_dir, legacy_fixture_dir):
    current = (source11_fixture_dir / "results_v2.dcr").read_text()
    legacy = (legacy_fixture_dir / "DCRexample.results.dcr").read_text()

    assert current != legacy

    current_first = read_dcr_results(source11_fixture_dir / "results_v2.dcr")[0]
    legacy_first = read_dcr_results(legacy_fixture_dir / "DCRexample.results.dcr")[0]
    assert current_first.numeric_values[3] == 21900
    assert legacy_first.numeric_values[3] == 21753
    assert not math.isclose(current_first.numeric_values[3], legacy_first.numeric_values[3])
