from __future__ import annotations

import numpy as np
import pytest

from bestpred.core.kernel import (
    LB_PER_KG,
    predict_records,
    predict_source11_mfp_multi_trait_debug,
    predict_source11_milk_305_debug,
    predict_source11_trait_305_debug,
)
from bestpred.core.source11 import simulate_source11_records
from bestpred.io.dcr import read_dcr_results
from bestpred.io.parameters import read_parameters
from bestpred.io.source11 import read_source11_examples

ALL_RECORD_INTEGERISH_FIELD_INDEXES = (
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
ALL_RECORD_TWO_DECIMAL_FIELD_INDEXES = (
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
    28,
    29,
    30,
    34,
    38,
    42,
)


def test_source11_milk_305_debug_prediction_builds_kernel_state(source11_fixture_dir) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    records = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )

    result = predict_source11_milk_305_debug(records[0], parameters)

    assert len(result.used_segments) == 9
    assert result.observation_covariance.shape == (9, 9)
    assert result.covariance_to_305.shape == (1, 9)
    assert result.deviations.shape == (9, 1)
    np.testing.assert_allclose(result.observation_covariance, result.observation_covariance.T)
    assert result.standard_305_variance > 0.0
    assert result.variance_factor > 0.0
    assert result.herd_ratio > 0.0
    assert result.milk_305_actual_output > 0.0


def test_source11_milk_305_first_deviation_uses_fortran_units_and_3x(source11_fixture_dir) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    records = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )

    result = predict_source11_milk_305_debug(records[0], parameters)
    first_segment = records[0].segments[0]
    observed_kg = first_segment.milk_yield / 10.0 / LB_PER_KG

    assert result.lactation_3x_factor == pytest.approx(1.0)
    assert result.deviations[0, 0] < observed_kg
    assert result.age_factor == pytest.approx(1.0078826592587493)
    assert result.deviations[0, 0] == pytest.approx(4.515531862261504)


def test_predict_records_emits_partial_source11_rows(source11_fixture_dir) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    records = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )

    rows = predict_records(records[:2], parameters)

    assert len(rows) == 2
    assert rows[0].animal_id == "HOUSA.EX.COW.0001"
    assert rows[0].fresh_date == "19990401"
    assert rows[0].dim == 255
    assert len(rows[0].numeric_values) == 43
    assert rows[0].numeric_values[0] > 0.0
    assert rows[0].numeric_values[1] > 0.0
    assert rows[0].numeric_values[2] > 0.0
    assert rows[0].numeric_values[3] == pytest.approx(21899.763049373112)
    assert rows[0].numeric_values[4] > 0.0
    assert rows[0].numeric_values[5] > 0.0
    assert rows[0].numeric_values[6] > 0.0
    assert rows[0].numeric_values[7] > rows[0].numeric_values[3]
    assert rows[0].numeric_values[11] == pytest.approx(rows[0].numeric_values[3])
    assert rows[0].numeric_values[15] > 0.0
    assert rows[0].numeric_values[15] < rows[0].numeric_values[3]
    assert rows[0].numeric_values[19] == pytest.approx(-0.06930241001145271)
    assert rows[0].numeric_values[20] == pytest.approx(0.03493231655888518)
    assert rows[0].numeric_values[21] == pytest.approx(-0.3133280320423313)
    assert rows[0].numeric_values[22] == pytest.approx(-0.5886838540047378)
    assert rows[0].numeric_values[23] == pytest.approx(0.9737063348316272)
    assert rows[0].numeric_values[24] == pytest.approx(0.9741747776847048)
    assert rows[0].numeric_values[25] == pytest.approx(0.9733911464458825)
    assert rows[0].numeric_values[26] == pytest.approx(0.9656630670598686)
    assert rows[0].numeric_values[27] == pytest.approx(0.8457026090457745)
    assert rows[0].numeric_values[28] == pytest.approx(0.8387685816117345)
    assert rows[0].numeric_values[29] == pytest.approx(0.8384122537762553)
    assert rows[0].numeric_values[30] == pytest.approx(0.7249518106029197)
    assert rows[0].numeric_values[31] == pytest.approx(21951.06366407857)
    assert rows[0].numeric_values[32] == pytest.approx(788.8927182794463)
    assert rows[0].numeric_values[33] == pytest.approx(697.1319330087377)
    assert rows[0].numeric_values[34] == pytest.approx(-0.00266239902915979)
    assert rows[0].numeric_values[35:39] == (20000.0, 700.0, 600.0, 3.08)
    assert rows[0].numeric_values[39] == pytest.approx(0.0)
    assert rows[0].numeric_values[40] == pytest.approx(0.0)
    assert rows[0].numeric_values[41] == pytest.approx(0.0)
    assert rows[0].numeric_values[42] == pytest.approx(0.28581611125620837)
    assert "nan" not in rows[0].raw_line


def test_source11_trait_305_debug_predicts_component_traits(source11_fixture_dir) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    records = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )

    fat = predict_source11_trait_305_debug(records[0], parameters, trait=2)
    protein = predict_source11_trait_305_debug(records[0], parameters, trait=3)
    scs = predict_source11_trait_305_debug(records[0], parameters, trait=4)

    assert fat.trait == 2
    assert protein.trait == 3
    assert scs.trait == 4
    assert fat.yld_305_actual_output > 0.0
    assert protein.yld_305_actual_output > 0.0
    assert scs.yld_305_actual_output > 0.0
    assert 0.0 < fat.reliability_305 <= 1.0
    assert fat.dcr_305 > 0.0
    assert fat.yld_365_actual_output > fat.yld_305_actual_output
    assert protein.yld_laclen_actual_output == pytest.approx(protein.yld_305_actual_output)
    assert fat.yld_partial_actual_output < fat.yld_305_actual_output
    assert fat.covariance_to_persistency.shape == (1, 9)
    assert fat.predicted_persistency == pytest.approx(0.034932316558851345)
    assert fat.persistency_reliability == pytest.approx(0.8387685816117345)
    assert fat.expanded_yield_output == pytest.approx(788.8927182794484)
    assert fat.herd_305_output == pytest.approx(700.0)
    assert fat.bumpiness == pytest.approx(6.677948149704135)
    assert scs.herd_305_output == pytest.approx(3.08)
    assert scs.bumpiness == pytest.approx(0.28581611125620837)
    assert fat.observation_covariance.shape == (9, 9)
    assert fat.covariance_to_365.shape == (1, 9)
    assert fat.covariance_to_laclen.shape == (1, 9)
    assert fat.covariance_to_partial.shape == (1, 9)
    assert protein.observation_covariance.shape == (9, 9)
    assert scs.observation_covariance.shape == (9, 9)


def test_source11_mfp_multi_trait_debug_builds_fortran_shaped_system(source11_fixture_dir) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    records = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )

    result = predict_source11_mfp_multi_trait_debug(records[0], parameters)

    assert len(result.used_observations) == 27
    assert result.observation_covariance.shape == (27, 27)
    assert result.deviations.shape == (27, 1)
    np.testing.assert_allclose(result.observation_covariance, result.observation_covariance.T)
    assert result.yld_305_outputs == pytest.approx(
        (21899.763049373112, 786.5970440676687, 694.5473636278799)
    )
    assert result.dcr_305 == pytest.approx(
        (101.63949215361454, 101.90112737287703, 101.6065914870441)
    )


def test_source11_milk_only_record_uses_fortran_missing_component_guards(
    source11_fixture_dir,
) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    records = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )
    python_row = predict_records(records[25:26], parameters)[0]
    oracle_row = read_dcr_results(source11_fixture_dir / "results_v2.dcr")[25]

    assert python_row.dim == 285
    assert python_row.numeric_values[1] == pytest.approx(oracle_row.numeric_values[1], abs=0.51)
    for index in (2, 6, 10, 14, 18, 22, 26, 30, 38, 42):
        assert python_row.numeric_values[index] == oracle_row.numeric_values[index]
    assert np.isnan(python_row.numeric_values[34])
    assert np.isnan(oracle_row.numeric_values[34])


@pytest.mark.golden
def test_source11_python_output_matches_current_oracle_for_ported_rounded_fields(
    source11_fixture_dir,
) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    records = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )
    python_row = predict_records(records[:1], parameters)[0]
    oracle_row = read_dcr_results(source11_fixture_dir / "results_v2.dcr")[0]

    integerish_field_indexes = (
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
        34,
        39,
        40,
        41,
    )
    for index in integerish_field_indexes:
        assert python_row.numeric_values[index] == pytest.approx(
            oracle_row.numeric_values[index],
            abs=0.51,
        )

    two_decimal_field_indexes = (
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
        38,
        42,
    )
    for index in two_decimal_field_indexes:
        assert python_row.numeric_values[index] == pytest.approx(
            oracle_row.numeric_values[index],
            abs=0.02,
        )

    assert python_row.numeric_values[35:38] == oracle_row.numeric_values[35:38]


@pytest.mark.golden
def test_source11_python_output_matches_current_oracle_for_all_records_stable_fields(
    source11_fixture_dir,
) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    records = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )
    python_rows = predict_records(records, parameters)
    oracle_rows = read_dcr_results(source11_fixture_dir / "results_v2.dcr")

    assert len(python_rows) == len(oracle_rows) == 43
    ported_row_pairs = [
        (python_row, oracle_row)
        for python_row, oracle_row in zip(python_rows, oracle_rows, strict=True)
        if not all(np.isnan(value) for value in python_row.numeric_values)
    ]
    assert len(ported_row_pairs) == 43

    for python_row, oracle_row in ported_row_pairs:
        assert python_row.fresh_date == oracle_row.fresh_date
        assert python_row.dim == oracle_row.dim
        for index in ALL_RECORD_INTEGERISH_FIELD_INDEXES:
            if np.isnan(python_row.numeric_values[index]) and np.isnan(
                oracle_row.numeric_values[index]
            ):
                continue
            assert python_row.numeric_values[index] == pytest.approx(
                oracle_row.numeric_values[index],
                abs=0.51,
            )
        for index in ALL_RECORD_TWO_DECIMAL_FIELD_INDEXES:
            if np.isnan(python_row.numeric_values[index]) and np.isnan(
                oracle_row.numeric_values[index]
            ):
                continue
            assert python_row.numeric_values[index] == pytest.approx(
                oracle_row.numeric_values[index],
                abs=0.02,
            )
