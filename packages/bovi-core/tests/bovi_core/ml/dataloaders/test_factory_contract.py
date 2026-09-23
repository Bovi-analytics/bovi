from collections.abc import Iterator
from typing import Any, ClassVar

import pytest
from bovi_core.ml.dataloaders import (
    AbstractDataLoader,
    DataLoaderConfig,
    DataLoaderFactory,
)
from bovi_core.ml.dataloaders import (
    create_dataloader as dispatch_dataloader,
)
from bovi_core.ml.models.config import ModelConfig
from bovi_core.ml.registry import DataLoaderFactoryRegistry


class ExampleDataLoaderConfig(DataLoaderConfig):
    model_key: ClassVar[str] = "example"


class ExampleModelConfig(ModelConfig):
    model_key: ClassVar[str] = "example"


class ExampleLoader(AbstractDataLoader):
    def __iter__(self) -> Iterator[Any]:
        return iter(())

    def __len__(self) -> int:
        return 0


def create_dataloader(
    data_config: ExampleDataLoaderConfig,
    model_config: ExampleModelConfig,
) -> AbstractDataLoader:
    del data_config, model_config
    return ExampleLoader.__new__(ExampleLoader)


def test_plain_function_satisfies_callable_factory_contract():
    factory: DataLoaderFactory[ExampleDataLoaderConfig, ExampleModelConfig] = create_dataloader

    assert factory is create_dataloader


def test_dispatch_delegates_to_registered_factory():
    data_config = ExampleDataLoaderConfig(split="train")
    model_config = ExampleModelConfig(framework="example")
    expected = create_dataloader(data_config, model_config)

    @DataLoaderFactoryRegistry.register("example")
    def factory(received_data_config, received_model_config):
        assert received_data_config is data_config
        assert received_model_config is model_config
        return expected

    try:
        result = dispatch_dataloader("example", data_config, model_config)
    finally:
        DataLoaderFactoryRegistry.clear()

    assert result is expected


def test_dispatch_rejects_invalid_factory_result():
    data_config = ExampleDataLoaderConfig(split="train")
    model_config = ExampleModelConfig(framework="example")

    @DataLoaderFactoryRegistry.register("invalid")
    def factory(received_data_config, received_model_config):
        del received_data_config, received_model_config
        return object()

    try:
        with pytest.raises(TypeError, match="expected AbstractDataLoader"):
            dispatch_dataloader("invalid", data_config, model_config)
    finally:
        DataLoaderFactoryRegistry.clear()
