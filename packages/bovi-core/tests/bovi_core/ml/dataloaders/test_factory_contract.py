from collections.abc import Iterator
from typing import Any, ClassVar

from bovi_core.ml.dataloaders import AbstractDataLoader, DataLoaderConfig, DataLoaderFactory
from bovi_core.ml.models.config import ModelConfig


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
