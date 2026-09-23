"""CPU inference wrapper around one Keras dense layer."""

import tensorflow as tf
from bovi_core.ml import Model

from .config import TensorFlowLinearModelConfig


class TensorFlowLinearModel(Model[tf.keras.Sequential, TensorFlowLinearModelConfig]):
    def __call__(self, features):
        with tf.device("/CPU:0"):
            inputs = tf.convert_to_tensor(features, dtype=tf.float32)
            return self.native_model(inputs, training=False).numpy().reshape(-1)
