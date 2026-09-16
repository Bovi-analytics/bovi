"""Build the scikit regression pipeline from typed split configuration."""

from __future__ import annotations

from bovi_core.ml.dataloaders import SklearnDataLoader, TransformRegistry
from bovi_core.ml.dataloaders.sources import TransformedSource

from scikit_sgd.models import ScikitSGDModelConfig

from .config import ScikitSGDDataLoaderConfig
from .dataset import ScikitRegressionDataset
from .source import RegressionJSONSource


def create_dataloader(
    data_config: ScikitSGDDataLoaderConfig,
    model_config: ScikitSGDModelConfig,
) -> SklearnDataLoader:
    """Build the source, transforms, dataset, and loader for one split."""
    source = RegressionJSONSource(data_config.source.path)
    transforms = TransformRegistry.from_config(
        [transform.model_dump() for transform in data_config.transforms]
    )
    transformed_source = TransformedSource(source, transforms)
    dataset = ScikitRegressionDataset(
        source=transformed_source,
        feature_names=model_config.feature_names,
        target_name=data_config.dataset.target_name,
    )
    return SklearnDataLoader(
        dataset=dataset,
        split=data_config.split,
        batch_size=data_config.dataloader.batch_size,
        shuffle=data_config.dataloader.shuffle,
        seed=data_config.dataloader.seed,
    )
