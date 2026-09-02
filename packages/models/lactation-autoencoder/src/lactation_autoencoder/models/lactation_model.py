"""Runtime wrapper for the lactation TensorFlow SavedModel."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import tensorflow as tf
from bovi_core.ml import Model

from .model_config import LactationAutoencoderModelConfig

ServingSignature = Callable[..., object]


class LactationAutoencoderModel(Model[tf.Module, LactationAutoencoderModelConfig]):
    """Loaded TensorFlow runtime state without storage or prediction concerns."""

    def __init__(
        self,
        native_model: tf.Module,
        config: LactationAutoencoderModelConfig,
        serving_signature: ServingSignature,
    ) -> None:
        super().__init__(native_model=native_model, config=config)
        self.serving_signature = serving_signature

    def __call__(self, *args: object, **kwargs: object) -> Any:
        """Invoke the configured TensorFlow serving signature."""
        return self.serving_signature(*args, **kwargs)
