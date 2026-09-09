"""Config-driven construction of the scikit regression data pipeline."""

from __future__ import annotations

from pathlib import Path

from bovi_core.config import Config, config_node_to_data
from bovi_core.ml.dataloaders import SklearnDataLoader, TransformRegistry
from bovi_core.ml.dataloaders.config import TabularDataSettings
from bovi_core.ml.dataloaders.sources import TransformedSource

from scikit_sgd.models import ScikitSGDModelConfig

from .dataset import ScikitRegressionDataset
from .source import RegressionJSONSource


def create_dataloader(
    config: Config,
    model_config: ScikitSGDModelConfig,
    split: str,
) -> SklearnDataLoader:
    """Build the source, transforms, dataset, and loader for one split."""
    node = config.experiment.models.scikit_sgd
    settings = TabularDataSettings.model_validate(
        config_node_to_data({"dataset": node.dataset, "split": getattr(node.dataloaders, split)})
    )
    split_config = settings.split
    source_path = Path(split_config.source.path)
    if not source_path.is_absolute():
        source_path = Path(config.project.project_root) / source_path

    source = RegressionJSONSource(source_path)
    transforms = TransformRegistry.from_config(
        [transform.model_dump() for transform in split_config.transforms]
    )
    transformed_source = TransformedSource(source, transforms)
    dataset = ScikitRegressionDataset(
        source=transformed_source,
        feature_names=model_config.feature_names,
        target_name=settings.dataset.target_name,
        config=config,
    )
    return SklearnDataLoader(
        dataset=dataset,
        config=config,
        split=split,
        model_name="scikit_sgd",
        batch_size=split_config.dataloader.batch_size,
        shuffle=split_config.dataloader.shuffle,
        seed=split_config.dataloader.seed,
    )
