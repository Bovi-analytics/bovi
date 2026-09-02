from __future__ import annotations

import math

import pytest

from bestpred.core.kernel import predict_pcdart_projected_actual_305, predict_records
from bestpred.io.dcr import read_dcr_results
from bestpred.io.parameters import read_parameters
from bestpred.io.pcdart import write_pcdart_output
from bestpred.io.source14 import read_source24_records

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


def test_read_source24_records_flattens_all_source14_files(source24_fixture_dir) -> None:
    records = read_source24_records(source24_fixture_dir / "pcdart_files.txt")

    assert len(records) == 6
    assert records[0].cow_id.strip() == "H0   1000001"
    assert records[2].compatibility_tag == "source14_eof_zero"
    assert records[3].cow_id.strip() == "H0   2000001"
    assert records[5].compatibility_tag == "source14_eof_zero"


@pytest.mark.golden
def test_source24_python_output_matches_aggregated_source14_oracle_numerics(
    source24_fixture_dir,
) -> None:
    parameters = read_parameters(source24_fixture_dir / "bestpred.par")
    records = read_source24_records(source24_fixture_dir / "pcdart_files.txt")
    oracle_rows = read_dcr_results(source24_fixture_dir / "results_v2.dcr")

    python_rows = predict_records(records, parameters, source11_compat=False)

    assert len(python_rows) == len(oracle_rows) == 6

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
def test_source24_pcdart_output_matches_current_oracle(source24_fixture_dir, tmp_path) -> None:
    parameters = read_parameters(source24_fixture_dir / "bestpred.par")
    records = read_source24_records(source24_fixture_dir / "pcdart_files.txt")
    rows = predict_records(records, parameters, source11_compat=False)
    projected_actuals = predict_pcdart_projected_actual_305(records, parameters)
    output = tmp_path / "pcdart.bpo"

    write_pcdart_output(
        output,
        records=records,
        rows=rows,
        projected_actuals=projected_actuals,
        include_compatibility_rows=False,
    )

    assert output.read_text(encoding="utf-8") == (source24_fixture_dir / "pcdart.bpo").read_text(
        encoding="utf-8"
    )
