"""Artifact loading for the lactation TensorFlow SavedModel."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import tensorflow as tf
from bovi_core.ml import ModelProviderRegistry, ResolvedModelArtifact

from .config import LactationAutoencoderModelConfig
from .model import LactationAutoencoderModel, ServingSignature

TENSORFLOW_SAVED_MODEL_FORMAT = "tensorflow_saved_model"


class LactationAutoencoderModelProvider:
    """Load resolved TensorFlow SavedModel artifacts.

    This package currently has no Python architecture factory or trainable
    checkpoint codec, so the provider intentionally exposes only artifact loading.
    """

    def load_artifact(
        self,
        config: LactationAutoencoderModelConfig,
        artifact: ResolvedModelArtifact[object],
    ) -> LactationAutoencoderModel:
        if artifact.format != TENSORFLOW_SAVED_MODEL_FORMAT:
            raise ValueError(
                f"Unsupported lactation model artifact format: {artifact.format!r}. "
                f"Expected {TENSORFLOW_SAVED_MODEL_FORMAT!r}."
            )

        native_model = artifact.payload
        if native_model is None:
            native_model = self._load_local_saved_model(artifact.local_path)

        signatures = getattr(native_model, "signatures", None)
        if signatures is None or config.signature_name not in signatures:
            raise ValueError(
                f"TensorFlow SavedModel does not expose signature {config.signature_name!r}."
            )

        signature = cast(ServingSignature, signatures[config.signature_name])
        return LactationAutoencoderModel(
            native_model=cast(tf.Module, native_model),
            config=config,
            serving_signature=signature,
        )

    @staticmethod
    def _load_local_saved_model(local_path: Path | None) -> object:
        if local_path is None:
            raise ValueError("A local TensorFlow SavedModel path or payload is required.")
        if not local_path.is_dir() or not (local_path / "saved_model.pb").is_file():
            raise ValueError(
                f"Invalid TensorFlow SavedModel directory: {local_path}. "
                "Expected a directory containing saved_model.pb."
            )
        try:
            return tf.saved_model.load(str(local_path))
        except Exception as exc:
            raise ValueError(
                f"Could not load TensorFlow SavedModel from {local_path}: {exc}"
            ) from exc


ModelProviderRegistry.register("autoencoder")(LactationAutoencoderModelProvider)
