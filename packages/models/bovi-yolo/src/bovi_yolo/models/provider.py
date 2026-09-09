"""Construction and local loading provider for Ultralytics YOLO models."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import cast

from bovi_core.ml import (
    ModelProviderRegistry,
    ResolvedCheckpoint,
    ResolvedModelArtifact,
)
from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]

from .config import YOLOModelConfig
from .model import YOLOModel

ULTRALYTICS_PT_FORMAT = "ultralytics-pt"
ULTRALYTICS_RUNTIME_FORMAT = "ultralytics-runtime"


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
        return self._load_resolved(
            config,
            checkpoint.format,
            checkpoint.local_path,
            checkpoint.payload,
        )

    def load_artifact(
        self,
        config: YOLOModelConfig,
        artifact: ResolvedModelArtifact[object],
    ) -> YOLOModel:
        """Load a YOLO runtime from a local deployment artifact."""
        return self._load_resolved(
            config,
            artifact.format,
            artifact.local_path,
            artifact.payload,
        )

    def _load_resolved(
        self,
        config: YOLOModelConfig,
        resource_format: str,
        local_path: PathLike[str] | None,
        payload: object | None,
    ) -> YOLOModel:
        if payload is not None:
            if not callable(payload):
                raise TypeError("YOLO resource payload must be a callable native model")
            if resource_format not in {ULTRALYTICS_PT_FORMAT, ULTRALYTICS_RUNTIME_FORMAT}:
                raise ValueError(
                    f"Unsupported YOLO runtime format: {resource_format!r}. "
                    f"Expected {ULTRALYTICS_PT_FORMAT!r} or "
                    f"{ULTRALYTICS_RUNTIME_FORMAT!r}."
                )
            return self._wrap(cast(YOLO, payload), config)

        if local_path is None:
            raise ValueError("YOLO resource requires a resolved local path or native payload")
        if resource_format != ULTRALYTICS_PT_FORMAT:
            raise ValueError(
                f"Unsupported YOLO file format: {resource_format!r}. "
                f"Expected {ULTRALYTICS_PT_FORMAT!r}."
            )

        resolved_path = Path(local_path)
        if not resolved_path.is_file():
            raise ValueError(f"Resolved YOLO weights file does not exist: {resolved_path}")

        return self._wrap(YOLO(str(resolved_path), task=config.task), config)

    @staticmethod
    def _wrap(native_model: YOLO, config: YOLOModelConfig) -> YOLOModel:
        return YOLOModel(native_model=native_model, config=config)


ModelProviderRegistry.register("yolo")(YOLOModelProvider)
