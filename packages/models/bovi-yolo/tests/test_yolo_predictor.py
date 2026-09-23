"""Tests for the YOLO predictor."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest

if TYPE_CHECKING:
    from bovi_core.config import Config


def _create_model(native_model: MagicMock):
    from bovi_yolo.models import YOLOModel, YOLOModelConfig

    return YOLOModel(
        native_model=native_model,
        config=YOLOModelConfig(framework="pytorch"),
    )


class TestYOLOPredictorInitialization:
    def test_model_is_injected(self, yolo_config: Config) -> None:
        from bovi_yolo.predictors import YOLOPredictor

        model = _create_model(MagicMock())
        predictor = YOLOPredictor(model=model, config=yolo_config)

        assert predictor.model is model


class TestYOLOPredictorPredict:
    def test_raw_return_format(
        self,
        yolo_config: Config,
        sample_image: npt.NDArray[np.uint8],
    ) -> None:
        from bovi_yolo.predictors import YOLOPredictor

        native_model = MagicMock(return_value=[MagicMock()])
        predictor = YOLOPredictor(model=_create_model(native_model), config=yolo_config)

        result = predictor.predict(sample_image, return_format="raw")

        assert result == native_model.return_value

    def test_rich_return_format(
        self,
        yolo_config: Config,
        sample_image: npt.NDArray[np.uint8],
    ) -> None:
        from bovi_yolo.predictors import YOLOPredictor
        from bovi_yolo.predictors.results import YoloPredictionResult

        mock_result = MagicMock()
        mock_result.orig_img = sample_image
        mock_result.boxes.xyxy.cpu().numpy.return_value = np.array([[10, 10, 100, 100]])
        mock_result.boxes.cls.cpu().numpy.return_value = np.array([0])
        mock_result.boxes.conf.cpu().numpy.return_value = np.array([0.95])
        mock_result.masks = None
        mock_result.names = {0: "cow"}

        native_model = MagicMock(return_value=[mock_result])
        native_model.names = {0: "cow"}
        predictor = YOLOPredictor(model=_create_model(native_model), config=yolo_config)

        result = predictor.predict(sample_image, return_format="rich")

        assert isinstance(result, YoloPredictionResult)
        assert result.num_predictions == 1

    def test_base_return_format(
        self,
        yolo_config: Config,
        sample_image: npt.NDArray[np.uint8],
    ) -> None:
        from bovi_yolo.predictors import YOLOPredictor

        mock_result = MagicMock()
        mock_result.orig_img = sample_image
        mock_result.boxes.xyxy.cpu().numpy.return_value = np.array([[10, 10, 100, 100]])
        mock_result.boxes.cls.cpu().numpy.return_value = np.array([0])
        mock_result.boxes.conf.cpu().numpy.return_value = np.array([0.95])
        mock_result.masks = None
        mock_result.names = {0: "cow"}

        native_model = MagicMock(return_value=[mock_result])
        native_model.names = {0: "cow"}
        predictor = YOLOPredictor(model=_create_model(native_model), config=yolo_config)

        result = predictor.predict(sample_image, return_format="base")

        assert isinstance(result, dict)
        assert "boxes_xyxy" in result
        assert "num_predictions" in result

    def test_prediction_error_wraps_exception(
        self,
        yolo_config: Config,
        sample_image: npt.NDArray[np.uint8],
    ) -> None:
        from bovi_yolo.predictors import PredictionError, YOLOPredictor

        native_model = MagicMock(side_effect=RuntimeError("GPU error"))
        predictor = YOLOPredictor(model=_create_model(native_model), config=yolo_config)

        with pytest.raises(PredictionError, match="YOLO prediction failed"):
            predictor.predict(sample_image, return_format="raw")


class TestPredictionError:
    def test_prediction_error_attributes(self) -> None:
        from bovi_yolo.predictors import PredictionError

        original = RuntimeError("test")
        error = PredictionError("Failed", "yolo", original)

        assert error.model_name == "yolo"
        assert error.original_exception is original
        assert str(error) == "Failed"
