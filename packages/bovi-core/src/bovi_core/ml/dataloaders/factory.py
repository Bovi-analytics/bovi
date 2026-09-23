"""Callable contract for model-owned dataloader factories."""

from typing import Protocol, TypeVar

from bovi_core.ml.dataloaders.config import DataLoaderConfig
from bovi_core.ml.dataloaders.loaders.base_loader import AbstractDataLoader
from bovi_core.ml.models.config import ModelConfig

DataLoaderConfigT = TypeVar(
    "DataLoaderConfigT",
    bound=DataLoaderConfig,
    contravariant=True,
)
ModelConfigT = TypeVar("ModelConfigT", bound=ModelConfig, contravariant=True)


class DataLoaderFactory(Protocol[DataLoaderConfigT, ModelConfigT]):
    """A model-owned function that assembles one concrete dataloader."""

    def __call__(
        self,
        data_config: DataLoaderConfigT,
        model_config: ModelConfigT,
    ) -> AbstractDataLoader: ...


def create_dataloader(
    model_key: str,
    data_config: DataLoaderConfig,
    model_config: ModelConfig,
) -> AbstractDataLoader:
    """Discover the model package's factory and delegate construction to it."""
    from bovi_core.ml.registry import DataLoaderFactoryRegistry

    loader = DataLoaderFactoryRegistry.create(model_key, data_config, model_config)
    if not isinstance(loader, AbstractDataLoader):
        raise TypeError(
            f"Dataloader factory '{model_key}' returned {type(loader).__name__}; "
            "expected AbstractDataLoader"
        )
    return loader
