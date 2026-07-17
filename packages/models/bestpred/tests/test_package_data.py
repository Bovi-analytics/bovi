from __future__ import annotations

from importlib import resources

import pytest

from bestpred.core.age import load_aiplage_data
from bestpred.core.curves import load_regional_curve_tables
from bestpred.core.scs import load_scs_adjustment_data


@pytest.mark.parametrize(
    "resource_name",
    [
        "aiplage.h",
        "adjust.scs",
        "bestpred.f90",
    ],
)
def test_bundled_fortran_resources_are_available(resource_name: str) -> None:
    resource = resources.files("bestpred.data").joinpath(resource_name)

    assert resource.is_file()
    assert resource.read_text(encoding="utf-8")


def test_default_loaders_use_bundled_resources() -> None:
    load_aiplage_data.cache_clear()
    load_scs_adjustment_data.cache_clear()
    load_regional_curve_tables.cache_clear()

    aiplage = load_aiplage_data()
    scs = load_scs_adjustment_data()
    curves = load_regional_curve_tables()

    assert aiplage.region.shape == (97, 4)
    assert aiplage.floats["hmequ0"].shape == (5, 12, 6, 12)
    assert scs.dim.shape == (306, 2, 2)
    assert curves["regional_woods_means"].shape == (3, 3, 2, 7)
    assert curves["seasonal_mandg_sd"].shape == (4, 1, 2, 4, 7)
