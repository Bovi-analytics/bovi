from datetime import date
from pathlib import Path
from typing import ClassVar
from unittest.mock import Mock

import pytest
import yaml
from bovi_core.config import ConfigNode, config_node_to_data
from bovi_core.ml.models.config import ModelConfig
from bovi_core.ml.trainers.config import EvaluationConfig, TrainingConfig
from pydantic import BaseModel, ConfigDict, ValidationError


class NestedSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int = 1


class ModelSettings(ModelConfig):
    model_key: ClassVar[str] = "strict_config_test"
    framework: str = "example"
    nested: NestedSettings = NestedSettings()
    steps: list[NestedSettings] = []


class TrainingSettings(TrainingConfig):
    model_key: ClassVar[str] = "strict_config_test"
    epochs: int = 2
    nested: NestedSettings = NestedSettings()
    steps: list[NestedSettings] = []


class EvaluationSettings(EvaluationConfig):
    model_key: ClassVar[str] = "strict_config_test"
    nested: NestedSettings = NestedSettings()
    steps: list[NestedSettings] = []


@pytest.mark.parametrize("schema", [ModelSettings, TrainingSettings, EvaluationSettings])
@pytest.mark.parametrize(
    ("values", "location"),
    [
        ({"typo": 9}, ("typo",)),
        ({"nested": {"count": 2, "coutn": 9}}, ("nested", "coutn")),
        ({"steps": [{"count": 2, "coutn": 9}]}, ("steps", 0, "coutn")),
    ],
)
def test_unknown_fields_match_for_kwargs_and_nodes(schema, values, location):
    for validate in (lambda: schema(**values), lambda: schema.model_validate(ConfigNode(values))):
        with pytest.raises(ValidationError) as error:
            validate()
        assert (location, "extra_forbidden") in [
            (issue["loc"], issue["type"]) for issue in error.value.errors()
        ]


@pytest.mark.parametrize(
    ("schema", "section"),
    [
        (ModelSettings, "architecture"),
        (TrainingSettings, "training"),
        (EvaluationSettings, "evaluation"),
    ],
)
@pytest.mark.parametrize("unknown", [False, True])
def test_from_config_validates_selected_yaml_section(config_setup, schema, section, unknown):
    values = yaml.safe_load("nested:\n  count: 3\nsteps:\n  - count: 4\n")
    if unknown:
        values["steps"][0]["coutn"] = 8
    model = ConfigNode({"framework": "example", section: values})
    model.unrelated_client = object()
    setattr(config_setup.experiment.models, "strict_config_test", model)

    if unknown:
        with pytest.raises(ValidationError) as error:
            schema.from_config(config_setup)
        assert error.value.errors()[0]["loc"] == ("steps", 0, "coutn")
    else:
        assert schema.from_config(config_setup) == schema(**values)


def test_extractor_recurses_without_mutation_or_resolving_secrets():
    manager = Mock()
    node = ConfigNode({"nested": {"steps": [{"value": ConfigNode({"count": 2})}]}}, manager)
    data = config_node_to_data(node)
    assert data == {"nested": {"steps": [{"value": {"count": 2}}]}}
    nested = data["nested"]
    assert isinstance(nested, dict)
    steps = nested["steps"]
    assert isinstance(steps, list)
    steps.append({})
    assert len(node.nested.steps) == 1
    manager.get_secret.assert_not_called()


@pytest.mark.parametrize(
    "value",
    [
        object(),
        Mock(),
        ConfigNode({"token": "secret-value"}, is_secrets=True),
        {"secrets": {"token": "secret-value"}},
    ],
)
def test_extractor_rejects_unsafe_nested_values_without_exposing_them(value):
    with pytest.raises(ValueError) as error:
        config_node_to_data(ConfigNode({"steps": [{"value": value}]}))
    assert "secret-value" not in str(error.value)


def test_extractor_rejects_whole_config_and_cycles(config_setup):
    with pytest.raises(ValueError, match="whole Config"):
        config_node_to_data(config_setup)
    node = ConfigNode({})
    node.loop = [node]
    with pytest.raises(ValueError, match="cycles"):
        config_node_to_data(node)


def test_extractor_does_not_resolve_secret_node():
    manager = Mock()
    node = ConfigNode({"token": "secret-reference"}, manager, is_secrets=True)
    with pytest.raises(ValueError, match="secrets"):
        config_node_to_data(node)
    manager.get_secret.assert_not_called()


@pytest.mark.parametrize("schema", [ModelSettings, TrainingSettings, EvaluationSettings])
def test_valid_direct_node_matches_kwargs_and_preserves_typed_kwargs(schema):
    values = {"nested": {"count": 3}, "steps": [{"count": 4}]}
    assert schema.model_validate(ConfigNode(values)) == schema(**values)
    assert schema(nested=NestedSettings(count=3)).nested.count == 3
    assert schema.model_validate(schema()) == schema()


def test_extractor_preserves_data_types_and_repeated_references():
    shared = ConfigNode({"count": 2})
    values = {
        "path": Path("weights.bin"),
        "date": date(2026, 9, 9),
        "values": (None, True, 1, 2.5, b"data"),
        "nodes": [shared, shared],
    }
    result = config_node_to_data(values)
    assert result == {**values, "nodes": [{"count": 2}, {"count": 2}]}
    nodes = result["nodes"]
    assert isinstance(nodes, list)
    assert nodes[0] is not nodes[1]


def test_extractor_rejects_nonstring_dictionary_keys():
    with pytest.raises(ValueError, match="string keys"):
        config_node_to_data({"nested": [{1: "value"}]})
