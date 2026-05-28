"""Tests for public characteristics module exports."""

from types import ModuleType

from lactationcurve import characteristics


def test_characteristics_submodules_are_public_exports() -> None:
    """Keep pdoc navigation aware of the characteristics child modules."""
    expected_submodules = {
        "best_predict",
        "lactation_curve_characteristics",
        "method_test_interval",
    }

    assert expected_submodules.issubset(characteristics.__all__)
    for submodule_name in expected_submodules:
        assert isinstance(getattr(characteristics, submodule_name), ModuleType)
