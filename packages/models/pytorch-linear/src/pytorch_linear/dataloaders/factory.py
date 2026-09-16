"""Build the PyTorch linear pipeline from typed split configuration."""

from bovi_core.ml.dataloaders import PyTorchDataLoader, TransformRegistry
from bovi_core.ml.dataloaders.sources import TransformedSource

from ..models.config import PyTorchLinearModelConfig
from .config import PyTorchLinearDataLoaderConfig
from .dataset import LinearDataset
from .source import LinearJSONSource


def create_dataloader(
    data_config: PyTorchLinearDataLoaderConfig,
    model_config: PyTorchLinearModelConfig,
) -> PyTorchDataLoader:
    source = TransformedSource(
        LinearJSONSource(data_config.source.path),
        TransformRegistry.from_config(
            [transform.model_dump() for transform in data_config.transforms]
        ),
    )
    dataset = LinearDataset(source, model_config.feature_names, data_config.dataset.target_name)
    settings = data_config.dataloader
    return PyTorchDataLoader(
        dataset=dataset,
        split=data_config.split,
        batch_size=settings.batch_size,
        shuffle=settings.shuffle,
        seed=settings.seed,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
        drop_last=settings.drop_last,
        persistent_workers=settings.persistent_workers,
        prefetch_factor=settings.prefetch_factor,
    )
