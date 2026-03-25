"""Tests for YOLO predictor."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest

if TYPE_CHECKING:
    from bovi_core.config import Config


class TestYOLOPredictorInitialization:
    def test_initialization(self, yolo_config: Config) -> None:
        """Test predictor can be initialized."""
        from bovi_yolo.predictors import YOLOPredictor

        predictor = YOLOPredictor(config=yolo_config)
        assert predictor.model_instance is None

    def test_model_instance_not_set_raises(
        self, yolo_config: Config
    ) -> None:
        """Test predict raises when model not set."""
        from bovi_yolo.predictors import PredictionError, YOLOPredictor

        predictor = YOLOPredictor(config=yolo_config)

        with pytest.raises(PredictionError, match="Model instance not set"):
            predictor.predict(
                np.zeros((100, 100, 3), dtype=np.uint8),
                return_format="raw",
            )


class TestYOLOPredictorSetModel:
    def test_set_model_instance(self, yolo_config: Config) -> None:
        """Test set_model_instance stores model."""
        from bovi_yolo.predictors import YOLOPredictor

        predictor = YOLOPredictor(config=yolo_config)

        mock_model = MagicMock()
        predictor.set_model_instance(mock_model)
        assert predictor.model_instance is mock_model


class TestYOLOPredictorPredict:
    def test_raw_return_format(
        self,
        yolo_config: Config,
        sample_image: npt.NDArray[np.uint8],
    ) -> None:
        """Test raw return format returns model output directly."""
        from bovi_yolo.predictors import YOLOPredictor

        predictor = YOLOPredictor(config=yolo_config)

        mock_model = MagicMock()
        mock_results = [MagicMock()]
        mock_model.return_value = mock_results
        predictor.set_model_instance(mock_model)

        result = predictor.predict(sample_image, return_format="raw")
        assert result == mock_results

    def test_rich_return_format(
        self,
        yolo_config: Config,
        sample_image: npt.NDArray[np.uint8],
    ) -> None:
        """Test rich return format returns YoloPredictionResult."""
        from bovi_yolo.predictors import YOLOPredictor
        from bovi_yolo.predictors.results import YoloPredictionResult

        predictor = YOLOPredictor(config=yolo_config)

        # Create mock ultralytics result
        mock_result = MagicMock()
        mock_result.orig_img = sample_image
        mock_boxes = MagicMock()
        mock_boxes.xyxy.cpu().numpy.return_value = np.array(
            [[10, 10, 100, 100]]
        )
        mock_boxes.cls.cpu().numpy.return_value = np.array([0])
        mock_boxes.conf.cpu().numpy.return_value = np.array([0.95])
        mock_result.boxes = mock_boxes
        mock_result.masks = None
        mock_result.names = {0: "cow"}

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        mock_model.names = {0: "cow"}
        predictor.set_model_instance(mock_model)

        result = predictor.predict(sample_image, return_format="rich")
        assert isinstance(result, YoloPredictionResult)
        assert result.num_predictions == 1

    def test_base_return_format(
        self,
        yolo_config: Config,
        sample_image: npt.NDArray[np.uint8],
    ) -> None:
        """Test base return format returns serializable dict."""
        from bovi_yolo.predictors import YOLOPredictor

        predictor = YOLOPredictor(config=yolo_config)

        mock_result = MagicMock()
        mock_result.orig_img = sample_image
        mock_boxes = MagicMock()
        mock_boxes.xyxy.cpu().numpy.return_value = np.array(
            [[10, 10, 100, 100]]
        )
        mock_boxes.cls.cpu().numpy.return_value = np.array([0])
        mock_boxes.conf.cpu().numpy.return_value = np.array([0.95])
        mock_result.boxes = mock_boxes
        mock_result.masks = None
        mock_result.names = {0: "cow"}

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        mock_model.names = {0: "cow"}
        predictor.set_model_instance(mock_model)

        result = predictor.predict(sample_image, return_format="base")
        assert isinstance(result, dict)
        assert "boxes_xyxy" in result
        assert "num_predictions" in result

    def test_prediction_error_wraps_exception(
        self,
        yolo_config: Config,
        sample_image: npt.NDArray[np.uint8],
    ) -> None:
        """Test PredictionError wraps underlying exceptions."""
        from bovi_yolo.predictors import PredictionError, YOLOPredictor

        predictor = YOLOPredictor(config=yolo_config)

        mock_model = MagicMock()
        mock_model.side_effect = RuntimeError("GPU error")
        predictor.set_model_instance(mock_model)

        with pytest.raises(PredictionError, match="YOLO prediction failed"):
            predictor.predict(sample_image, return_format="raw")


class TestPredictionError:
    def test_prediction_error_attributes(self) -> None:
        """Test PredictionError stores model_name and original_exception."""
        from bovi_yolo.predictors import PredictionError

        original = RuntimeError("test")
        error = PredictionError("Failed", "yolo", original)

        assert error.model_name == "yolo"
        assert error.original_exception is original
        assert str(error) == "Failed"
