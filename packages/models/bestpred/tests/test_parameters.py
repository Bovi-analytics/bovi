from __future__ import annotations

from bestpred.io.parameters import read_parameters
from bestpred.models import BestpredSource, BreedCode


def test_read_parameters_source11_fixture(source11_fixture_dir):
    parameters = read_parameters(source11_fixture_dir / "bestpred.par")

    assert parameters.source == BestpredSource.DCR_EXAMPLE
    assert parameters.source11_breed == BreedCode.HOLSTEIN
    assert parameters.write_curve == 0
    assert parameters.write_data == 0
    assert parameters.curve_single == 0
    assert parameters.onscreen == 0
    assert parameters.maxshow == 0
    assert parameters.dim0 == (115, 115, 150, 155, 161, 152, 159, 148)
