"""Tensor conversion and reconstruction metrics for lactation batches."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import sqrt
from typing import Any

import numpy as np
import tensorflow as tf

from lactation_autoencoder.models import LactationAutoencoderModel


def batch_to_tensors(batch: Mapping[str, Any]) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
    features = batch["features"]
    if not isinstance(features, Mapping):
        raise TypeError("Lactation batch features must be a mapping")

    milk = tf.convert_to_tensor(features["milk"], dtype=tf.float32)
    if milk.shape.rank == 2:
        milk = tf.expand_dims(milk, axis=-1)
    inputs = {
        "input_11": milk,
        "input_12": tf.convert_to_tensor(features["parity"], dtype=tf.float32),
        "input_13": tf.convert_to_tensor(features["events"], dtype=tf.int32),
        "input_15": tf.convert_to_tensor(features["herd_stats"], dtype=tf.float32),
    }
    labels = tf.convert_to_tensor(batch["labels"], dtype=tf.float32)
    return inputs, labels


def measure(
    model: LactationAutoencoderModel, batches: Iterable[Mapping[str, Any]]
) -> tuple[int, dict[str, float]]:
    examples = elements = 0
    squared_error = absolute_error = 0.0
    for batch in batches:
        inputs, expected = batch_to_tensors(batch)
        predicted = model.trainable_model(inputs, training=False)
        expected_array = np.asarray(expected)
        error = np.asarray(predicted) - expected_array
        if not error.size or not np.isfinite(error).all():
            raise ValueError("Lactation predictions must be finite and nonempty")
        examples += expected_array.shape[0]
        elements += int(error.size)
        squared_error += float(np.sum(error**2))
        absolute_error += float(np.sum(np.abs(error)))
    if not examples or not elements:
        raise ValueError("Cannot evaluate an empty dataloader")
    mse = squared_error / elements
    return examples, {"mse": mse, "mae": absolute_error / elements, "rmse": sqrt(mse)}
