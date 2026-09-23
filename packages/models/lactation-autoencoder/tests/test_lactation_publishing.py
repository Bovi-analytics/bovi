"""Exercise serving with real TensorFlow artifacts and local MLflow roundtrips."""

from types import SimpleNamespace

import mlflow.pyfunc
import numpy as np
import pytest
import tensorflow as tf
from bovi_core.ml.publishing.wrappers import TensorFlowSavedModelWrapper


class NamedWrapper(TensorFlowSavedModelWrapper):
    def get_input_name_mapping(self):
        return {
            "milk": "input_11",
            "parity": "input_12",
            "events": "input_13",
            "herd_stats": "input_15",
        }


@pytest.fixture
def saved_model(tmp_path):
    # Small deterministic artifact with the same input layout as the real model.
    class NativeModel(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec([None, 4, 1], tf.float32, name="input_11"),
                tf.TensorSpec([None, 1], tf.float32, name="input_12"),
                tf.TensorSpec([None, 4], tf.int32, name="input_13"),
                tf.TensorSpec([None, 2], tf.float32, name="input_15"),
            ]
        )
        def serve(self, input_11, input_12, input_13, input_15):
            return {
                "prediction": (
                    input_11[..., 0]
                    + input_12
                    + tf.cast(input_13, tf.float32)
                    + tf.reduce_sum(input_15, axis=1, keepdims=True)
                )
            }

    native = NativeModel()
    path = tmp_path / "native"
    tf.saved_model.save(native, str(path), signatures={"serving_default": native.serve})
    return path


@pytest.fixture
def inputs():
    return {
        "milk": np.arange(8, dtype=np.float32).reshape(2, 4),
        "parity": np.array([[1], [2]], dtype=np.float32),
        "events": np.zeros((2, 4), dtype=np.int32),
        "herd_stats": np.ones((2, 2), dtype=np.float32),
    }


def test_core_wrapper_accepts_named_batches_and_signature_dtypes(saved_model, inputs):
    wrapper = NamedWrapper()
    wrapper.load_context(SimpleNamespace(artifacts={"model_path": str(saved_model)}))
    batch = {**inputs, "milk": inputs["milk"][..., None]}
    result = wrapper.predict(None, batch)
    np.testing.assert_allclose(result["prediction"], inputs["milk"] + inputs["parity"] + 2)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_lactation_local_mlflow_roundtrip(saved_model, inputs, tmp_path, batch_size):
    from lactation_autoencoder.publishing import LactationSavedModelWrapper

    batch = {key: value[:batch_size] for key, value in inputs.items()}
    wrapper = LactationSavedModelWrapper()
    wrapper.load_context(SimpleNamespace(artifacts={"model_path": str(saved_model)}))
    expected = wrapper.predict(None, batch)
    np.testing.assert_allclose(expected["prediction"], batch["milk"] + batch["parity"] + 2)
    signature = mlflow.models.infer_signature(batch, expected)
    destination = tmp_path / "mlflow-model"
    # Serialize a fresh wrapper, not the loaded TensorFlow graph.
    mlflow.pyfunc.save_model(
        path=str(destination),
        python_model=LactationSavedModelWrapper(),
        artifacts={"model_path": str(saved_model)},
        input_example=batch,
        signature=signature,
        pip_requirements=["tensorflow", "numpy", "bovi-core", "lactation-autoencoder"],
    )
    loaded = mlflow.pyfunc.load_model(str(destination))
    actual = loaded.predict(batch)
    assert actual.keys() == expected.keys()
    np.testing.assert_allclose(actual["prediction"], expected["prediction"])
    assert actual["prediction"].shape == (batch_size, 4)
    # Neither shaping nor MLflow serialization may mutate the caller's inputs.
    assert batch["milk"].shape == (batch_size, 4)


@pytest.mark.parametrize("invalid", ["missing", "extra", "unbatched", "mismatched", "empty"])
def test_lactation_rejects_invalid_batches(saved_model, inputs, invalid):
    from lactation_autoencoder.publishing import LactationSavedModelWrapper

    wrapper = LactationSavedModelWrapper()
    wrapper.load_context(SimpleNamespace(artifacts={"model_path": str(saved_model)}))
    if invalid == "missing":
        inputs.pop("events")
    elif invalid == "extra":
        inputs["unknown"] = np.ones((2, 1))
    elif invalid == "unbatched":
        inputs["milk"] = inputs["milk"][0]
    elif invalid == "mismatched":
        inputs["parity"] = inputs["parity"][:1]
    else:
        inputs = {key: value[:0] for key, value in inputs.items()}
    with pytest.raises(ValueError):
        wrapper.predict(None, inputs)
