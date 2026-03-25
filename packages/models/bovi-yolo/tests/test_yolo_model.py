"""Tests for YOLO model."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import torch

if TYPE_CHECKING:
    from bovi_core.config import Config


class TestYOLOModelTypes:
    def test_set_model_types(self) -> None:
        """Test _set_model_types sets model_name and device."""
        from bovi_yolo.models import YOLOModel

        model = YOLOModel.__new__(YOLOModel)
        model._set_model_types()
        assert model.model_name == "yolo"
        assert isinstance(model.device, torch.device)
        assert model.device.type in ("cpu", "cuda")


class TestYOLOModelLoading:
    def test_load_model_invalid_location_raises(self) -> None:
        """Test unsupported weights_location raises ValueError."""
        from bovi_yolo.models import YOLOModel

        model = YOLOModel.__new__(YOLOModel)
        model.weights_location = "invalid"
        model.weights_path = "fake.pt"

        with pytest.raises(ValueError, match="Unsupported weights_location"):
            model.load_model()

    def test_call_without_model_raises(self) -> None:
        """Test __call__ raises when model not loaded."""
        from bovi_yolo.models import YOLOModel

        model = YOLOModel.__new__(YOLOModel)
        model.model = None

        with pytest.raises(RuntimeError, match="Model not loaded"):
            model("test_input")


class TestYOLOModelFromConfig:
    def test_from_config_requires_predictor(self, yolo_config: Config) -> None:
        """Test from_config raises ValueError when predictor is None."""
        from bovi_yolo.models import YOLOModel

        with pytest.raises(ValueError, match="predictor is required"):
            YOLOModel.from_config(config=yolo_config)

    @patch("bovi_yolo.models.yolo_model.YOLOModel.load_model")
    def test_from_config_with_predictor(self, mock_load: MagicMock, yolo_config: Config) -> None:
        """Test from_config creates model when predictor is provided."""
        from bovi_yolo.models import YOLOModel
        from bovi_yolo.predictors import YOLOPredictor

        mock_load.return_value = MagicMock()

        predictor = YOLOPredictor(config=yolo_config)
        model = YOLOModel.from_config(config=yolo_config, predictor=predictor)

        assert isinstance(model, YOLOModel)
        assert model.predictor is predictor
        mock_load.assert_called_once()

    @patch("bovi_yolo.models.yolo_model.YOLOModel.load_model")
    def test_from_config_custom_weights_path(
        self, mock_load: MagicMock, yolo_config: Config
    ) -> None:
        """Test from_config uses custom weights_path when provided."""
        from bovi_yolo.models import YOLOModel
        from bovi_yolo.predictors import YOLOPredictor

        mock_load.return_value = MagicMock()

        predictor = YOLOPredictor(config=yolo_config)
        model = YOLOModel.from_config(
            config=yolo_config,
            predictor=predictor,
            weights_path="/custom/path/yolo.pt",
        )

        assert model.weights_path == "/custom/path/yolo.pt"

    @patch("bovi_yolo.models.yolo_model.YOLOModel.load_model")
    def test_from_config_reads_weights_location_from_config(
        self, mock_load: MagicMock, yolo_config: Config
    ) -> None:
        """Test from_config uses weights_location from config."""
        from bovi_yolo.models import YOLOModel
        from bovi_yolo.predictors import YOLOPredictor

        mock_load.return_value = MagicMock()

        predictor = YOLOPredictor(config=yolo_config)
        model = YOLOModel.from_config(config=yolo_config, predictor=predictor)

        assert model.weights_location == "local"


class TestYOLOModelRegistry:
    def test_registered_in_model_registry(self) -> None:
        """Test YOLOModel is registered via @ModelRegistry.register."""
        from bovi_core.ml import ModelRegistry

        # Import to trigger registration
        import bovi_yolo  # noqa: F401

        assert "yolo" in ModelRegistry._models


class TestGetDevice:
    def test_get_device_returns_device(self) -> None:
        """Test get_device returns a torch.device."""
        from bovi_yolo.models.yolo_model import get_device

        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cpu", "cuda")
