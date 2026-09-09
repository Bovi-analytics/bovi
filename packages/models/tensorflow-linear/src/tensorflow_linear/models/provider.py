"""Create a CPU regressor or load a compiled Keras checkpoint."""

import json

import tensorflow as tf
from bovi_core.ml import ModelProviderRegistry, ResolvedCheckpoint, ResolvedModelArtifact

from .config import TensorFlowLinearModelConfig
from .model import TensorFlowLinearModel

FORMAT = "tensorflow-linear-keras"


class TensorFlowLinearModelProvider:
    def create(self, config: TensorFlowLinearModelConfig) -> TensorFlowLinearModel:
        with tf.device("/CPU:0"):
            native = tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(len(config.feature_names),)),
                    tf.keras.layers.Dense(1, kernel_initializer="zeros", bias_initializer="zeros"),
                ]
            )
            native.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss="mse")
        return TensorFlowLinearModel(native_model=native, config=config)

    def restore_checkpoint(
        self, config: TensorFlowLinearModelConfig, checkpoint: ResolvedCheckpoint[object]
    ) -> TensorFlowLinearModel:
        return self._load(config, checkpoint)

    def load_artifact(
        self, config: TensorFlowLinearModelConfig, artifact: ResolvedModelArtifact[object]
    ) -> TensorFlowLinearModel:
        return self._load(config, artifact)

    def _load(self, config, resource):
        if resource.format != FORMAT or resource.local_path is None:
            raise ValueError("Expected a local tensorflow-linear-keras checkpoint")
        names = json.loads(resource.local_path.with_suffix(".json").read_text())
        if tuple(names) != config.feature_names:
            raise ValueError("Checkpoint feature order differs from model config")
        with tf.device("/CPU:0"):
            native = tf.keras.models.load_model(resource.local_path, safe_mode=True)
        if not isinstance(native, tf.keras.Sequential):
            raise TypeError("Expected a Sequential linear model")
        return TensorFlowLinearModel(native_model=native, config=config)


ModelProviderRegistry.register("tensorflow_linear")(TensorFlowLinearModelProvider)
