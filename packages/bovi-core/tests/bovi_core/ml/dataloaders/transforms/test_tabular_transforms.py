"""Tests for generic stateless tabular transforms."""

import numpy as np
import pytest
from bovi_core.ml.dataloaders.transforms import (
    NumericClipTransform,
    NumericScaleTransform,
    TransformRegistry,
)


def test_numeric_clip_transforms_nested_configured_fields_without_mutating_input():
    data = {
        "features": {"milk": np.array([-2.0, 20.0, 70.0]), "days": 400},
        "metadata": {"farm_id": "farm-a"},
    }
    transform = NumericClipTransform(ranges={"milk": [0.0, 60.0], "days": [1.0, 305.0]})

    result = transform(data)

    np.testing.assert_array_equal(result["features"]["milk"], np.array([0.0, 20.0, 60.0]))
    assert result["features"]["days"] == 305.0
    assert result["metadata"] == {"farm_id": "farm-a"}
    np.testing.assert_array_equal(data["features"]["milk"], np.array([-2.0, 20.0, 70.0]))


def test_numeric_scale_transforms_nested_fields_and_leaves_other_values_unchanged():
    data = {
        "features": {"milk": [15.0, 30.0], "days": 152.5, "breed": "HF"},
        "labels": 12.0,
    }
    transform = NumericScaleTransform(factors={"milk": 60.0, "days": 305.0})

    result = transform(data)

    np.testing.assert_allclose(result["features"]["milk"], np.array([0.25, 0.5]))
    assert result["features"]["days"] == pytest.approx(0.5)
    assert result["features"]["breed"] == "HF"
    assert result["labels"] == 12.0


def test_tabular_transforms_are_registered_and_validate_parameters():
    assert isinstance(
        TransformRegistry.create("numeric_clip", ranges={"milk": [0.0, 60.0]}),
        NumericClipTransform,
    )
    assert isinstance(
        TransformRegistry.create("numeric_scale", factors={"milk": 60.0}),
        NumericScaleTransform,
    )

    with pytest.raises(ValueError, match="exactly two bounds"):
        NumericClipTransform(ranges={"milk": [0.0]})
    with pytest.raises(ValueError, match="non-zero"):
        NumericScaleTransform(factors={"milk": 0.0})
