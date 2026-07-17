from __future__ import annotations

import math

import pytest

from bestpred.core.kernel import predict_records
from bestpred.io.dcr import read_dcr_results
from bestpred.io.parameters import read_parameters
from bestpred.io.source10 import read_source10_records

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


def test_read_source10_records_emits_current_fortran_two_row_flow(source10_fixture_dir) -> None:
    records = read_source10_records(source10_fixture_dir / "format4.dat")

    assert len(records) == 2
    assert records[0].cow_id == "HOUSA00093WNM4798"
    assert records[0].fresh_date == "19940117"
    assert records[0].length == 290
    assert records[0].segments == ()
    assert records[0].herd_me_milk == 27505
    assert records[0].herd_me_fat == 1004
    assert records[0].herd_me_protein == 0

    assert len(records[1].segments) == 7
    assert records[1].segments[0].dim == 11
    assert records[1].segments[0].times_milked == 2
    assert records[1].segments[0].times_weighed == 2
    assert records[1].segments[0].times_sampled == 2
    assert records[1].segments[0].milk_yield == 605
    assert records[1].segments[0].fat_percent == 49
    assert records[1].segments[0].protein_percent == 89
    assert records[1].segments[0].scs == 31
    assert records[1].segments[-1].dim == 192


@pytest.mark.golden
def test_source10_python_output_matches_current_oracle_numerics(source10_fixture_dir) -> None:
    parameters = read_parameters(source10_fixture_dir / "bestpred.par")
    records = read_source10_records(source10_fixture_dir / "format4.dat")
    oracle_rows = read_dcr_results(source10_fixture_dir / "results_v2.dcr")

    python_rows = predict_records(records, parameters, source11_compat=False)

    assert len(python_rows) == len(oracle_rows) == 2
    assert python_rows[1].animal_id == "HOUSA00093WNM4798"
    assert python_rows[1].fresh_date == "19940117"
    assert python_rows[1].dim == 192

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
