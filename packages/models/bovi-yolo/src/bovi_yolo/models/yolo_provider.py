"""Construction and local loading provider for Ultralytics YOLO models."""

from __future__ import annotations

from os import PathLike
from typing import cast

from bovi_core.ml import (
    ModelProviderRegistry,
    ResolvedCheckpoint,
    ResolvedModelArtifact,
)
from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]

from .yolo_config import YOLOModelConfig
from .yolo_model import YOLOModel


class YOLOModelProvider:
    """Create YOLO runtimes from definitions or already-resolved resources."""

    def create(self, config: YOLOModelConfig) -> YOLOModel:
        """Create a fresh model from an Ultralytics architecture definition."""
        return self._wrap(YOLO(config.model_source, task=config.task), config)

    def restore_checkpoint(
        self,
        config: YOLOModelConfig,
        checkpoint: ResolvedCheckpoint[object],
    ) -> YOLOModel:
        """Restore a YOLO runtime from a local training checkpoint."""
        return self._load_resolved(config, checkpoint.local_path, checkpoint.payload)

    def load_artifact(
        self,
        config: YOLOModelConfig,
        artifact: ResolvedModelArtifact[object],
    ) -> YOLOModel:
        """Load a YOLO runtime from a local deployment artifact."""
        return self._load_resolved(config, artifact.local_path, artifact.payload)

    def _load_resolved(
        self,
        config: YOLOModelConfig,
        local_path: PathLike[str] | None,
        payload: object | None,
    ) -> YOLOModel:
        if payload is not None:
            if not callable(payload):
                raise TypeError("YOLO resource payload must be a callable native model")
            return self._wrap(cast(YOLO, payload), config)

        if local_path is None:
            raise ValueError("YOLO resource requires a resolved local path or native payload")

        return self._wrap(YOLO(str(local_path), task=config.task), config)

    @staticmethod
    def _wrap(native_model: YOLO, config: YOLOModelConfig) -> YOLOModel:
        return YOLOModel(native_model=native_model, config=config)


ModelProviderRegistry.register("yolo")(YOLOModelProvider)
