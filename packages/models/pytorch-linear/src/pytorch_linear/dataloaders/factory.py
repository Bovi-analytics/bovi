"""Build the model-specific data pipeline from the existing Bovi YAML."""

from pathlib import Path

from bovi_core.config import Config
from bovi_core.ml.dataloaders import SklearnDataLoader, TransformRegistry
from bovi_core.ml.dataloaders.sources import TransformedSource

from ..models.config import PyTorchLinearModelConfig
from .dataset import LinearDataset
from .source import LinearJSONSource


def create_dataloader(
    config: Config, model_config: PyTorchLinearModelConfig, split: str
) -> SklearnDataLoader:
    node = getattr(config.experiment.models, "pytorch_linear")
    settings = getattr(node.dataloaders, split)
    if settings.source.type != "json_records":
        raise ValueError("Only json_records sources are supported")
    path = Path(settings.source.path)
    if not path.is_absolute():
        path = Path(config.project.project_root) / path
    source = TransformedSource(
        LinearJSONSource(path),
        list(TransformRegistry.from_config(getattr(settings, "transforms", [])).values()),
    )
    dataset = LinearDataset(source, model_config.feature_names, node.dataset.target_name)
    # The shared NumPy batcher keeps data preparation independent of the training framework.
    return SklearnDataLoader(dataset, config, split, model_name="pytorch_linear")
