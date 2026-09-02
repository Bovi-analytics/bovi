from __future__ import annotations

import pytest

from bestpred.core.age import aiplage_factors, format4_age_factors, load_aiplage_data, perpetual_day
from bestpred.core.source11 import simulate_source11_records
from bestpred.io.parameters import read_parameters
from bestpred.io.source11 import read_source11_examples


def test_load_aiplage_data_parses_original_header() -> None:
    data = load_aiplage_data()

    assert data.region.shape == (97, 4)
    assert data.floats["hmregtm"].shape == (12, 5)
    assert data.floats["hmequ0"].shape == (5, 12, 6, 12)


def test_aiplage_matches_c_for_source11_first_record() -> None:
    factors = aiplage_factors(
        breed="H",
        age_at_freshening_months=72,
        fresh_year=1999,
        fresh_month=4,
        parity=5,
        state=35,
        previous_days_open=200,
    )

    assert factors.milk == pytest.approx(1.004666388471)
    assert factors.fat == pytest.approx(1.002309482628)
    assert factors.protein == pytest.approx(1.014795718361)


def test_aiplage_matches_c_for_holstein_first_lactation() -> None:
    factors = aiplage_factors(
        breed="H",
        age_at_freshening_months=36,
        fresh_year=1999,
        fresh_month=4,
        parity=1,
        state=35,
        previous_days_open=0,
    )

    assert factors.milk == pytest.approx(1.204229208829)
    assert factors.fat == pytest.approx(1.190092715050)
    assert factors.protein == pytest.approx(1.181066143205)


def test_aiplage_agebase_36_matches_c_second_pass() -> None:
    factors = aiplage_factors(
        breed="H",
        age_at_freshening_months=72,
        fresh_year=1999,
        fresh_month=4,
        parity=5,
        state=35,
        previous_days_open=200,
        agebase=36,
    )

    assert factors.milk == pytest.approx(0.886935614807)
    assert factors.fat == pytest.approx(0.883814383487)
    assert factors.protein == pytest.approx(0.920216391810)


def test_aiplage_defaults_unknown_breed_to_holstein() -> None:
    holstein = aiplage_factors(
        breed="H",
        age_at_freshening_months=72,
        fresh_year=1999,
        fresh_month=4,
        parity=5,
        state=35,
        previous_days_open=200,
    )
    unknown = aiplage_factors(
        breed="X",
        age_at_freshening_months=72,
        fresh_year=1999,
        fresh_month=4,
        parity=5,
        state=35,
        previous_days_open=200,
    )

    assert unknown == holstein


def test_format4_age_factors_match_source11_record(source11_fixture_dir) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    records = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )

    first = records[0]
    factors = format4_age_factors(first)

    assert perpetual_day(first.fresh_date) == 14335
    assert perpetual_day(first.birth_date) == 12144
    assert factors.milk == pytest.approx(1.0078826592587493, abs=1e-7)
    assert factors.fat == pytest.approx(1.0169273540909634, abs=1e-7)
    assert factors.protein == pytest.approx(1.0117895339228578, abs=1e-7)
