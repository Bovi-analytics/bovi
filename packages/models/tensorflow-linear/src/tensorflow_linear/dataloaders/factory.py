"""Build the TensorFlow linear pipeline from typed split configuration."""

from bovi_core.ml.dataloaders import TensorFlowDataLoader, TransformRegistry
from bovi_core.ml.dataloaders.sources import TransformedSource

from ..models.config import TensorFlowLinearModelConfig
from .config import TensorFlowLinearDataLoaderConfig
from .dataset import LinearDataset
from .source import LinearJSONSource


def create_dataloader(
    data_config: TensorFlowLinearDataLoaderConfig,
    model_config: TensorFlowLinearModelConfig,
) -> TensorFlowDataLoader:
    source = TransformedSource(
        LinearJSONSource(data_config.source.path),
        TransformRegistry.from_config(
            [transform.model_dump() for transform in data_config.transforms]
        ),
    )
    dataset = LinearDataset(source, model_config.feature_names, data_config.dataset.target_name)
    settings = data_config.dataloader
    return TensorFlowDataLoader(
        dataset=dataset,
        split=data_config.split,
        batch_size=settings.batch_size,
        shuffle=settings.shuffle,
        seed=settings.seed,
        buffer_size=settings.buffer_size,
        prefetch_buffer_size=settings.prefetch_buffer_size,
        cache=settings.cache,
        reshuffle_each_iteration=settings.reshuffle_each_iteration,
    )
