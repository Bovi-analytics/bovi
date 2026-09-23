from typing import Any, Protocol, TypeVar

from .config import ModelConfig
from .model import Model
from .resources import ResolvedCheckpoint, ResolvedModelArtifact

ModelT = TypeVar("ModelT", bound=Model[Any, Any], covariant=True)
ModelConfigT = TypeVar("ModelConfigT", bound=ModelConfig, contravariant=True)


class ModelProvider(Protocol[ModelT, ModelConfigT]):
    """Capability for constructing a fresh runtime model."""

    def create(self, config: ModelConfigT) -> ModelT: ...


class CheckpointModelProvider(Protocol[ModelT, ModelConfigT]):
    """Capability for restoring resumable training state."""

    def restore_checkpoint(
        self,
        config: ModelConfigT,
        checkpoint: ResolvedCheckpoint[object],
    ) -> ModelT: ...


class ArtifactModelProvider(Protocol[ModelT, ModelConfigT]):
    """Capability for loading a deployment/evaluation artifact."""

    def load_artifact(
        self,
        config: ModelConfigT,
        artifact: ResolvedModelArtifact[object],
    ) -> ModelT: ...
