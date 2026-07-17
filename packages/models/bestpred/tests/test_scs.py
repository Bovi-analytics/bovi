from __future__ import annotations

import pytest

from bestpred.core.scs import (
    adjusted_scs,
    format4_scs_age_factor,
    load_scs_adjustment_data,
    scs_age_factor,
)
from bestpred.core.source11 import simulate_source11_records
from bestpred.io.parameters import read_parameters
from bestpred.io.source11 import read_source11_examples


def test_load_scs_adjustment_data_parses_original_file() -> None:
    data = load_scs_adjustment_data()

    assert data.dim.shape == (306, 2, 2)
    assert data.age.shape == (121, 2)
    assert data.month.shape == (13, 4, 2)
    assert data.dim[15, 0, 0] == pytest.approx(45.0)
    assert data.age[18, 0] == pytest.approx(1.23)


@pytest.mark.parametrize(
    ("breed", "parity", "fresh_month", "state", "age", "dim", "scs", "expected"),
    [
        ("H", 5, 4, 12, 71, 305, 329, 277),
        ("H", 1, 4, 35, 36, 305, 329, 355),
        ("J", 2, 12, 66, 17, 14, 329, 396),
        ("G", 3, 7, 75, 130, 400, 999, 902),
        ("B", 2, 1, 24, 50, 60, 1, 48),
    ],
)
def test_adjusted_scs_matches_active_c_routine(
    breed: str,
    parity: int,
    fresh_month: int,
    state: int,
    age: int,
    dim: int,
    scs: int,
    expected: int,
) -> None:
    assert (
        adjusted_scs(
            breed=breed,
            parity=parity,
            fresh_month=fresh_month,
            state=state,
            age_at_freshening_months=age,
            dim=dim,
            scs=scs,
        )
        == expected
    )


def test_scs_age_factor_matches_fortran_fmt4_ratio() -> None:
    assert scs_age_factor(
        breed="H",
        parity=5,
        fresh_month=4,
        state=12,
        age_at_freshening_months=71,
    ) == pytest.approx(277 / 329)


def test_format4_scs_age_factor_match_source11_record(source11_fixture_dir) -> None:
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")
    records = simulate_source11_records(
        read_source11_examples(source11_fixture_dir / "DCRexample.txt"),
        parameters,
    )

    assert format4_scs_age_factor(records[0]) == pytest.approx(277 / 329)
