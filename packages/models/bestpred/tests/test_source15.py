from __future__ import annotations

import math

import pytest

from bestpred.core.kernel import predict_records
from bestpred.io.dcr import read_dcr_results
from bestpred.io.parameters import read_parameters
from bestpred.io.source15 import read_source15_records

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


def test_read_source15_records_applies_means_override_only_to_detail_row(
    source15_fixture_dir,
) -> None:
    records = read_source15_records(
        source15_fixture_dir / "format4.dat",
        source15_fixture_dir / "format4.means",
    )

    assert len(records) == 2

    assert records[0].segments == ()
    assert records[0].herd_me_milk == 27505
    assert records[0].herd_me_fat == 1004
    assert records[0].herd_me_protein == 0
    assert records[0].herd_me_scs == 0.0

    assert len(records[1].segments) == 7
    assert records[1].length == 192
    assert records[1].herd_me_milk == 24000
    assert records[1].herd_me_fat == 900
    assert records[1].herd_me_protein == 700
    assert records[1].herd_me_scs == pytest.approx(3.5)


def test_read_source15_records_zeroes_detail_means_on_id_mismatch(tmp_path, source15_fixture_dir):
    means_path = tmp_path / "format4.means"
    means_path.write_text("WRONGCOWID0000000 19940117 24000  900  700  350\n", encoding="utf-8")

    records = read_source15_records(source15_fixture_dir / "format4.dat", means_path)

    assert records[1].herd_me_milk == 0
    assert records[1].herd_me_fat == 0
    assert records[1].herd_me_protein == 0
    assert records[1].herd_me_scs == 0.0


@pytest.mark.golden
def test_source15_python_output_matches_current_oracle_numerics(source15_fixture_dir) -> None:
    parameters = read_parameters(source15_fixture_dir / "bestpred.par")
    records = read_source15_records(
        source15_fixture_dir / "format4.dat",
        source15_fixture_dir / "format4.means",
    )
    oracle_rows = read_dcr_results(source15_fixture_dir / "results_v2.dcr")

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
