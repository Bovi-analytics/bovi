"""Tests for model-provider, predictor, and dataloader-factory discovery."""

from unittest.mock import MagicMock, patch

import pytest
from bovi_core.ml import (
    DataLoaderFactoryRegistry,
    ModelProviderRegistry,
    PredictorRegistry,
)


@pytest.fixture(autouse=True)
def clear_registries():
    ModelProviderRegistry.clear()
    PredictorRegistry.clear()
    DataLoaderFactoryRegistry.clear()
    yield
    ModelProviderRegistry.clear()
    PredictorRegistry.clear()
    DataLoaderFactoryRegistry.clear()


def test_provider_registry_registers_and_constructs_provider():
    @ModelProviderRegistry.register("example")
    class ExampleProvider:
        def __init__(self, value: str) -> None:
            self.value = value

    provider = ModelProviderRegistry.create("example", value="configured")

    assert isinstance(provider, ExampleProvider)
    assert getattr(provider, "value") == "configured"
    assert ModelProviderRegistry.list_providers() == {
        "example": f"{ExampleProvider.__module__}.{ExampleProvider.__name__}"
    }


def test_predictor_registry_registers_and_constructs_predictor():
    @PredictorRegistry.register("example")
    class ExamplePredictor:
        def __init__(self, model: object) -> None:
            self.model = model

    model = object()
    predictor = PredictorRegistry.create("example", model=model)

    assert isinstance(predictor, ExamplePredictor)
    assert getattr(predictor, "model") is model


def test_missing_provider_reports_entry_point_group():
    with patch("bovi_core.ml.registry.entry_points", return_value=[]):
        with pytest.raises(ValueError, match="bovi.model_providers"):
            ModelProviderRegistry.get("missing")


@patch("bovi_core.ml.registry.entry_points")
def test_provider_is_discovered_lazily(mock_entry_points):
    entry_point = MagicMock()
    entry_point.name = "discovered"
    entry_point.value = "example.providers:ExampleProvider"

    class ExampleProvider:
        pass

    entry_point.load.return_value = ExampleProvider
    mock_entry_points.return_value = [entry_point]

    assert ModelProviderRegistry.get("discovered") is ExampleProvider
    entry_point.load.assert_called_once_with()


@patch("bovi_core.ml.registry.entry_points")
def test_failed_discovery_is_cached(mock_entry_points):
    mock_entry_points.return_value = []

    for _ in range(2):
        with pytest.raises(ValueError):
            ModelProviderRegistry.get("missing")

    discovery_calls = [
        call for call in mock_entry_points.call_args_list if call.kwargs.get("name") == "missing"
    ]
    assert len(discovery_calls) == 1


@patch("bovi_core.ml.registry.entry_points")
def test_list_available_does_not_load_plugins(mock_entry_points):
    entry_point = MagicMock()
    entry_point.name = "available"
    entry_point.value = "example.providers:ExampleProvider"
    mock_entry_points.return_value = [entry_point]

    assert ModelProviderRegistry.list_available() == {
        "available": "example.providers:ExampleProvider"
    }
    entry_point.load.assert_not_called()


def test_dataloader_factory_registry_registers_and_dispatches_function():
    data_config = object()
    model_config = object()
    expected_loader = object()

    @DataLoaderFactoryRegistry.register("example")
    def create_loader(received_data_config, received_model_config):
        assert received_data_config is data_config
        assert received_model_config is model_config
        return expected_loader

    assert DataLoaderFactoryRegistry.create("example", data_config, model_config) is expected_loader
    assert DataLoaderFactoryRegistry.list_factories() == {
        "example": f"{create_loader.__module__}.{create_loader.__name__}"
    }


@patch("bovi_core.ml.registry.entry_points")
def test_dataloader_factory_is_discovered_lazily(mock_entry_points):
    entry_point = MagicMock()
    entry_point.name = "discovered"
    entry_point.value = "example.dataloaders:create_dataloader"

    def create_loader(data_config, model_config):
        return data_config, model_config

    entry_point.load.return_value = create_loader
    mock_entry_points.return_value = [entry_point]

    assert DataLoaderFactoryRegistry.get("discovered") is create_loader
    entry_point.load.assert_called_once_with()


@patch("bovi_core.ml.registry.entry_points")
def test_dataloader_registry_rejects_non_callable_entry_point(mock_entry_points):
    entry_point = MagicMock()
    entry_point.name = "invalid"
    entry_point.value = "example.dataloaders:not_a_factory"
    entry_point.load.return_value = object()
    mock_entry_points.return_value = [entry_point]

    with pytest.raises(ValueError, match="bovi.dataloader_factories"):
        DataLoaderFactoryRegistry.get("invalid")
