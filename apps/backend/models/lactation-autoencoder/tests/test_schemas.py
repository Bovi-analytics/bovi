"""Schema validation tests that do not require loading model weights."""

import pytest
from pydantic import ValidationError
from schemas import AutoencoderBatchRequest, AutoencoderPredictRequest


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
