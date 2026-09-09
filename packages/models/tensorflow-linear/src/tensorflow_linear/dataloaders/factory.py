"""Build the model-specific data pipeline from the existing Bovi YAML."""

from pathlib import Path

from bovi_core.config import Config, config_node_to_data
from bovi_core.ml.dataloaders import TensorFlowDataLoader, TransformRegistry
from bovi_core.ml.dataloaders.config import TabularDataSettings
from bovi_core.ml.dataloaders.sources import TransformedSource

from ..models.config import TensorFlowLinearModelConfig
from .dataset import LinearDataset
from .source import LinearJSONSource


def create_dataloader(
    config: Config, model_config: TensorFlowLinearModelConfig, split: str
) -> TensorFlowDataLoader:
    node = getattr(config.experiment.models, "tensorflow_linear")
    data_settings = TabularDataSettings.model_validate(
        config_node_to_data({"dataset": node.dataset, "split": getattr(node.dataloaders, split)})
    )
    settings = data_settings.split
    path = Path(settings.source.path)
    if not path.is_absolute():
        path = Path(config.project.project_root) / path
    source = TransformedSource(
        LinearJSONSource(path),
        TransformRegistry.from_config(
            [transform.model_dump() for transform in settings.transforms]
        ),
    )
    dataset = LinearDataset(source, model_config.feature_names, data_settings.dataset.target_name)
    return TensorFlowDataLoader(
        dataset,
        config,
        split,
        model_name="tensorflow_linear",
        prefetch_buffer_size=1,
        batch_size=settings.dataloader.batch_size,
        shuffle=settings.dataloader.shuffle,
        seed=settings.dataloader.seed,
    )
