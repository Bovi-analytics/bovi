"""Config-driven construction of the lactation data pipeline."""

from __future__ import annotations

from bovi_core.ml.dataloaders import SklearnDataLoader, TransformRegistry
from bovi_core.ml.dataloaders.sources import TransformedSource

from lactation_autoencoder.models import LactationAutoencoderModelConfig

from .config import LactationAutoencoderDataLoaderConfig
from .dataset import LactationDataset
from .source import LactationJSONSource


def create_dataloader(
    data_config: LactationAutoencoderDataLoaderConfig,
    model_config: LactationAutoencoderModelConfig,
) -> SklearnDataLoader:
    """Build the source, transforms, dataset, and NumPy loader for one split."""
    source = LactationJSONSource(
        json_root_dir=data_config.source.json_root_dir,
        file_pattern=data_config.source.file_pattern,
        keep_in_memory=data_config.dataset.keep_in_memory,
    )
    transforms = TransformRegistry.from_config(
        [transform.model_dump() for transform in data_config.transforms]
    )
    transformed_source = TransformedSource(source, transforms)
    dataset = LactationDataset(
        source=transformed_source,
        max_days=data_config.dataset.max_days,
    )

    # This loader is the framework-neutral NumPy batcher for the nested feature
    # mapping. TensorFlow conversion belongs to the model or a future nested TF loader.
    return SklearnDataLoader(
        dataset=dataset,
        split=data_config.split,
        batch_size=data_config.dataloader.batch_size,
        shuffle=data_config.dataloader.shuffle,
        seed=data_config.dataloader.seed,
    )
