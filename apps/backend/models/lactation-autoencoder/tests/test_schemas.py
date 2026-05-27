"""Schema validation tests that do not require loading model weights."""

import pytest
from pydantic import ValidationError
from schemas import (
    AutoencoderBatchRequest,
    AutoencoderPredictRequest,
    project_periodic_records_to_daily,
)


def test_project_periodic_records_to_daily_zero_fills_unobserved_days():
    projected = project_periodic_records_to_daily([1, 3, 304], [20.0, 25.0, 18.0])

    assert len(projected) == 304
    assert projected[0] == 20.0
    assert projected[1] == 0.0
    assert projected[2] == 25.0
    assert projected[303] == 18.0
    assert sum(1 for value in projected if value != 0.0) == 3


def test_predict_request_accepts_periodic_records():
    request = AutoencoderPredictRequest.model_validate(
        {
            "dim": [10, 40, 70],
            "milkrecordings": [30.0, 38.0, 35.0],
            "parity": 2,
        }
    )

    assert request.model_milk_input()[9] == 30.0
    assert request.model_milk_input()[39] == 38.0
    assert request.model_milk_input()[69] == 35.0


def test_predict_request_rejects_mixed_daily_and_periodic_records():
    with pytest.raises(ValidationError):
        AutoencoderPredictRequest.model_validate(
            {
                "milk": [25.0, None, 27.0],
                "dim": [1, 3],
                "milkrecordings": [25.0, 27.0],
            }
        )


def test_predict_request_rejects_partial_periodic_records():
    with pytest.raises(ValidationError):
        AutoencoderPredictRequest.model_validate({"dim": [1, 3]})


def test_predict_request_rejects_periodic_length_mismatch():
    with pytest.raises(ValidationError):
        AutoencoderPredictRequest.model_validate(
            {
                "dim": [1, 3],
                "milkrecordings": [25.0],
            }
        )


def test_predict_request_rejects_periodic_dim_outside_autoencoder_horizon():
    with pytest.raises(ValidationError):
        AutoencoderPredictRequest.model_validate(
            {
                "dim": [1, 305],
                "milkrecordings": [25.0, 27.0],
            }
        )


@pytest.mark.parametrize("imputation_method", ["forward_fill", "backward_fill", "linear"])
def test_predict_request_accepts_supported_imputation_methods(imputation_method: str):
    request = AutoencoderPredictRequest.model_validate(
        {
            "milk": [25.0, None, 27.0],
            "imputation_method": imputation_method,
        }
    )

    assert request.imputation_method == imputation_method


@pytest.mark.parametrize("imputation_method", ["zero", "mean"])
def test_predict_request_rejects_removed_imputation_methods(imputation_method: str):
    with pytest.raises(ValidationError):
        AutoencoderPredictRequest.model_validate(
            {
                "milk": [25.0, None, 27.0],
                "imputation_method": imputation_method,
            }
        )


@pytest.mark.parametrize("imputation_method", ["zero", "mean"])
def test_batch_request_rejects_removed_imputation_methods(imputation_method: str):
    with pytest.raises(ValidationError):
        AutoencoderBatchRequest.model_validate(
            {
                "items": [{"milk": [25.0, None, 27.0]}],
                "imputation_method": imputation_method,
            }
        )
