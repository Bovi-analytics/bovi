from __future__ import annotations

import math

import pytest

from bestpred.core.kernel import predict_pcdart_projected_actual_305, predict_records
from bestpred.io.dcr import read_dcr_results
from bestpred.io.parameters import read_parameters
from bestpred.io.pcdart import write_pcdart_output
from bestpred.io.source14 import read_source14_records

INTEGERISH_INDEXES = (
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    9,
    11,
    12,
    13,
    15,
    16,
    17,
    31,
    32,
    33,
    35,
    36,
    37,
    39,
    40,
    41,
)
TWO_DECIMAL_INDEXES = (
    6,
    10,
    14,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    34,
    38,
    42,
)


def _assert_pcdart_output_matches(actual: str, expected: str) -> None:
    actual_lines = actual.splitlines()
    expected_lines = expected.splitlines()
    assert len(actual_lines) == len(expected_lines)

    for line_number, (actual_line, expected_line) in enumerate(
        zip(actual_lines, expected_lines, strict=True),
        start=1,
    ):
        actual_parts = actual_line.split()
        expected_parts = expected_line.split()
        assert actual_parts[:3] == expected_parts[:3], line_number
        assert len(actual_parts) == len(expected_parts), line_number
        for actual_token, expected_token in zip(actual_parts[3:], expected_parts[3:], strict=True):
            assert float(actual_token) == pytest.approx(float(expected_token), abs=1.01), (
                line_number,
                actual_line,
                expected_line,
            )


def test_read_source14_records_parses_detail_rows_and_trailing_eof_artifact(
    source14_fixture_dir,
) -> None:
    records = read_source14_records(source14_fixture_dir / "test241.txt")

    assert len(records) == 3

    assert records[0].cow_id.strip() == "H0   1000001"
    assert records[0].fresh_date == "20051023"
    assert records[0].parity == 1
    assert records[0].length == 27
    assert len(records[0].segments) == 2
    assert records[0].segments[0].dim == 6
    assert records[0].segments[1].dim == 27
    assert records[0].herd_me_milk == 25000
    assert records[0].herd_me_fat == 900
    assert records[0].herd_me_protein == 750
    assert records[0].herd_me_scs == pytest.approx(0.30)

    assert records[1].cow_id.strip() == "H0   1000002"
    assert records[1].fresh_date == "20051002"
    assert records[1].parity == 8
    assert records[1].length == 48
    assert len(records[1].segments) == 2
    assert records[1].segments[0].times_sampled == 1
    assert records[1].segments[1].dim == 48

    assert records[2].compatibility_tag == "source14_eof_zero"
    assert records[2].length == 0
    assert records[2].segments == ()
    assert records[2].parity == 8


@pytest.mark.golden
def test_source14_python_output_matches_current_oracle_numerics(source14_fixture_dir) -> None:
    parameters = read_parameters(source14_fixture_dir / "bestpred.par")
    records = read_source14_records(source14_fixture_dir / "test241.txt")
    oracle_rows = read_dcr_results(source14_fixture_dir / "results_v2.dcr")

    python_rows = predict_records(records, parameters, source11_compat=False)

    assert len(python_rows) == len(oracle_rows) == 3

    for python_row, oracle_row in zip(python_rows, oracle_rows, strict=True):
        for index in INTEGERISH_INDEXES:
            python_value = python_row.numeric_values[index]
            oracle_value = oracle_row.numeric_values[index]
            if math.isnan(python_value) and math.isnan(oracle_value):
                continue
            assert python_value == pytest.approx(oracle_value, abs=0.51)

        for index in TWO_DECIMAL_INDEXES:
            python_value = python_row.numeric_values[index]
            oracle_value = oracle_row.numeric_values[index]
            if math.isnan(python_value) and math.isnan(oracle_value):
                continue
            assert python_value == pytest.approx(oracle_value, abs=0.02)


@pytest.mark.golden
def test_source14_pcdart_output_matches_current_oracle(source14_fixture_dir, tmp_path) -> None:
    parameters = read_parameters(source14_fixture_dir / "bestpred.par")
    records = read_source14_records(source14_fixture_dir / "test241.txt")
    rows = predict_records(records, parameters, source11_compat=False)
    projected_actuals = predict_pcdart_projected_actual_305(records, parameters)
    output = tmp_path / "pcdart.bpo"

    write_pcdart_output(
        output,
        records=records,
        rows=rows,
        projected_actuals=projected_actuals,
        include_compatibility_rows=True,
    )

    _assert_pcdart_output_matches(
        output.read_text(encoding="utf-8"),
        (source14_fixture_dir / "pcdart.bpo").read_text(encoding="utf-8"),
    )
