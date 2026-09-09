"""Config-driven construction of the lactation data pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bovi_core.config import Config
from bovi_core.ml.dataloaders import SklearnDataLoader, TransformRegistry
from bovi_core.ml.dataloaders.sources import TransformedSource

from .dataset import LactationDataset
from .source import LactationJSONSource


def create_dataloader(
    config: Config,
    split: str,
    **override_params: Any,
) -> SklearnDataLoader:
    """Build the source, transforms, dataset, and NumPy loader for one split."""
    model_config = config.experiment.models.autoencoder
    split_config = getattr(model_config.dataloaders, split)
    source_config = split_config.source
    if source_config.type != "lactation_json":
        raise ValueError(
            f"Unsupported lactation source type: {source_config.type!r}. Expected 'lactation_json'."
        )

    json_root_dir = Path(source_config.json_root_dir)
    if not json_root_dir.is_absolute():
        json_root_dir = Path(config.project.project_root) / json_root_dir

    source = LactationJSONSource(
        json_root_dir=json_root_dir,
        file_pattern=getattr(source_config, "file_pattern", "*.json"),
        keep_in_memory=bool(model_config.dataset.keep_in_memory),
    )
    transforms = TransformRegistry.from_config(split_config.transforms)
    transformed_source = TransformedSource(source, transforms)
    dataset = LactationDataset(
        source=transformed_source,
        config=config,
        max_days=int(model_config.dataset.max_days),
    )

    # This loader is the framework-neutral NumPy batcher for the nested feature
    # mapping. TensorFlow conversion belongs to the model or a future nested TF loader.
    return SklearnDataLoader(
        dataset=dataset,
        config=config,
        split=split,
        model_name="autoencoder",
        **override_params,
    )
