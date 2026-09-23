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


@pytest.mark.parametrize("include_clip, expected", [(False, 1.0), (True, 0.75)])
def test_configured_pipeline_preserves_repeated_transforms_and_order(include_clip, expected):
    from bovi_core.ml.dataloaders.sources import DictSource, TransformedSource

    specs = [{"name": "numeric_scale", "params": {"factors": {"x": 2}}}]
    if include_clip:
        specs.append({"name": "numeric_clip", "params": {"ranges": {"x": [0, 3]}}})
    specs.append({"name": "numeric_scale", "params": {"factors": {"x": 4}}})

    transforms = TransformRegistry.from_config(specs)
    source = TransformedSource(DictSource([{"x": 8}]), transforms)

    assert isinstance(transforms, list)
    assert len(transforms) == len(specs)
    assert transforms[0] is not transforms[-1]
    assert source.load_item(0)["x"] == expected


def test_from_config_accepts_empty_pipeline_and_omitted_params(monkeypatch):
    from unittest.mock import Mock

    create = Mock(side_effect=[object(), object()])
    monkeypatch.setattr(TransformRegistry, "create", create)

    assert TransformRegistry.from_config([]) == []
    assert len(TransformRegistry.from_config([{"name": "example"}, {"name": "example"}])) == 2
    assert create.call_count == 2
    create.assert_called_with("example")


@pytest.mark.parametrize("params", [None, [], "invalid", 0, False])
def test_from_config_rejects_invalid_params_without_constructing_transform(params, monkeypatch):
    from unittest.mock import Mock

    create = Mock()
    monkeypatch.setattr(TransformRegistry, "create", create)

    with pytest.raises(TypeError, match="Transform 'numeric_scale' params must be a dictionary"):
        TransformRegistry.from_config([{"name": "numeric_scale", "params": params}])

    create.assert_not_called()
