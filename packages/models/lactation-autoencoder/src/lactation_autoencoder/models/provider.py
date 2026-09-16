"""Construct trainable models and load lactation model resources."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import tensorflow as tf
from bovi_core.ml import (
    ModelProviderRegistry,
    ResolvedCheckpoint,
    ResolvedModelArtifact,
)

from .config import LactationAutoencoderModelConfig
from .model import LactationAutoencoderModel, ServingSignature

TENSORFLOW_SAVED_MODEL_FORMAT = "tensorflow_saved_model"
LACTATION_WEIGHTS_FORMAT = "lactation-autoencoder-keras-weights"


class LactationAutoencoderModelProvider:
    """Own native architecture construction and resource restoration."""

    def create(self, config: LactationAutoencoderModelConfig) -> LactationAutoencoderModel:
        native_model = self._build_keras_model(config)

        def serve(**inputs: object) -> dict[str, tf.Tensor]:
            return {"activation_2": native_model(inputs, training=False)}

        return LactationAutoencoderModel(native_model, config, serve)

    def restore_checkpoint(
        self,
        config: LactationAutoencoderModelConfig,
        checkpoint: ResolvedCheckpoint[object],
    ) -> LactationAutoencoderModel:
        """Restore weights into a fresh architecture and optimizer."""
        if checkpoint.format != LACTATION_WEIGHTS_FORMAT or checkpoint.local_path is None:
            raise ValueError(f"Expected a local {LACTATION_WEIGHTS_FORMAT!r} checkpoint")
        saved_config = checkpoint.metadata.get("model_config")
        if saved_config is not None and saved_config != config.model_dump(mode="json"):
            raise ValueError("Checkpoint model configuration differs from requested config")
        model = self.create(config)
        prefix = checkpoint.local_path.with_suffix("")
        tf.train.Checkpoint(model=model.trainable_model).restore(str(prefix)).expect_partial()
        return model

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

    @staticmethod
    def _build_keras_model(config: LactationAutoencoderModelConfig) -> tf.keras.Model:
        """Build the four-input reconstruction architecture on CPU."""
        with tf.device("/CPU:0"):
            milk = tf.keras.Input(shape=(config.input_dim, 1), dtype=tf.float32, name="input_11")
            parity = tf.keras.Input(shape=(1,), dtype=tf.float32, name="input_12")
            events = tf.keras.Input(shape=(config.input_dim,), dtype=tf.int32, name="input_13")
            herd_stats = tf.keras.Input(
                shape=(config.num_herd_stats,), dtype=tf.float32, name="input_15"
            )

            milk_features = tf.keras.layers.Flatten()(milk)
            event_features = tf.keras.layers.Embedding(
                input_dim=config.num_events,
                output_dim=min(8, config.latent_dim),
            )(events)
            event_features = tf.keras.layers.GlobalAveragePooling1D()(event_features)
            combined = tf.keras.layers.Concatenate()(
                [milk_features, event_features, parity, herd_stats]
            )
            encoded = tf.keras.layers.Dense(config.latent_dim, activation="relu", name="latent")(
                combined
            )
            reconstructed = tf.keras.layers.Dense(
                config.input_dim, activation="sigmoid", name="activation_2"
            )(encoded)
            native_model = tf.keras.Model(
                inputs={
                    "input_11": milk,
                    "input_12": parity,
                    "input_13": events,
                    "input_15": herd_stats,
                },
                outputs=reconstructed,
                name="lactation_autoencoder",
            )
            native_model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        return native_model


ModelProviderRegistry.register("autoencoder")(LactationAutoencoderModelProvider)
