"""Tests for YOLO prediction result.

Moved from bovi-core/tests/bovi_core/ml/predictors/results/test_yolo_result.py
with updated imports.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest


class TestYoloPredictionResultEmpty:
    def test_empty_result(self) -> None:
        """Test creating an empty prediction result."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        result = YoloPredictionResult()
        assert result.num_predictions == 0
        assert result.boxes is None
        assert result.masks is None
        assert result.scores is None
        assert result.class_ids is None
        assert result.class_names is None

    def test_empty_serializable(self) -> None:
        """Test serialization of empty result."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        result = YoloPredictionResult()
        serialized = result.to_serializable()
        assert serialized["boxes_xyxy"] is None
        assert serialized["num_predictions"] == 0

    def test_empty_get_crops(self) -> None:
        """Test get_crops on empty result."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        result = YoloPredictionResult()
        assert result.get_crops() == []


class TestYoloPredictionResultBasic:
    def test_basic_detection(
        self,
        sample_image: npt.NDArray[np.uint8],
        sample_boxes: npt.NDArray[np.float64],
        sample_scores: npt.NDArray[np.float64],
        sample_class_ids: npt.NDArray[np.float64],
        sample_class_names: list[str],
    ) -> None:
        """Test creating result with basic detection data."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        result = YoloPredictionResult(
            original_image=sample_image,
            boxes=sample_boxes,
            scores=sample_scores,
            class_ids=sample_class_ids,
            class_names=sample_class_names,
        )
        assert result.num_predictions == 2
        assert result.image_height == 480
        assert result.image_width == 640

    def test_num_predictions_from_masks(self) -> None:
        """Test num_predictions falls back to masks count."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        masks = [
            np.zeros((480, 640), dtype=np.uint8),
            np.zeros((480, 640), dtype=np.uint8),
        ]
        result = YoloPredictionResult(masks=masks)
        assert result.num_predictions == 2


class TestYoloPredictionResultBoxFormats:
    def test_xyxy_format(
        self, sample_boxes: npt.NDArray[np.float64]
    ) -> None:
        """Test xyxy box format returns input boxes."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        result = YoloPredictionResult(boxes=sample_boxes)
        xyxy = result.get_boxes_xyxy()
        assert np.array_equal(xyxy, sample_boxes)

    def test_xywh_format(
        self, sample_boxes: npt.NDArray[np.float64]
    ) -> None:
        """Test xywh conversion (center_x, center_y, width, height)."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        result = YoloPredictionResult(boxes=sample_boxes)
        xywh = result.get_boxes_xywh()

        # First box: [100, 50, 300, 250] -> center_x=200, center_y=150, w=200, h=200
        assert np.isclose(xywh[0, 0], 200.0)
        assert np.isclose(xywh[0, 1], 150.0)
        assert np.isclose(xywh[0, 2], 200.0)
        assert np.isclose(xywh[0, 3], 200.0)

    def test_ltwh_format(
        self, sample_boxes: npt.NDArray[np.float64]
    ) -> None:
        """Test ltwh conversion (left, top, width, height)."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        result = YoloPredictionResult(boxes=sample_boxes)
        ltwh = result.get_boxes_ltwh()

        # First box: [100, 50, 300, 250] -> left=100, top=50, w=200, h=200
        assert np.isclose(ltwh[0, 0], 100.0)
        assert np.isclose(ltwh[0, 1], 50.0)
        assert np.isclose(ltwh[0, 2], 200.0)
        assert np.isclose(ltwh[0, 3], 200.0)

    def test_empty_boxes_returns_empty(self) -> None:
        """Test box format methods return empty array for no boxes."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        result = YoloPredictionResult()
        assert len(result.get_boxes_xyxy()) == 0
        assert len(result.get_boxes_xywh()) == 0
        assert len(result.get_boxes_ltwh()) == 0


class TestYoloPredictionResultSerialization:
    def test_to_serializable(
        self,
        sample_image: npt.NDArray[np.uint8],
        sample_boxes: npt.NDArray[np.float64],
        sample_scores: npt.NDArray[np.float64],
        sample_class_ids: npt.NDArray[np.float64],
        sample_class_names: list[str],
    ) -> None:
        """Test to_serializable produces valid dict."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        result = YoloPredictionResult(
            original_image=sample_image,
            boxes=sample_boxes,
            scores=sample_scores,
            class_ids=sample_class_ids,
            class_names=sample_class_names,
        )
        serialized = result.to_serializable()

        assert serialized["boxes_xyxy"] is not None
        assert serialized["num_predictions"] == 2
        assert serialized["scores"] is not None
        assert serialized["class_names"] == ["cow", "cow"]
        assert serialized["metadata"]["box_format"] == "xyxy"

    def test_to_human(
        self,
        sample_image: npt.NDArray[np.uint8],
        sample_boxes: npt.NDArray[np.float64],
        sample_scores: npt.NDArray[np.float64],
    ) -> None:
        """Test to_human includes all box formats."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        result = YoloPredictionResult(
            original_image=sample_image,
            boxes=sample_boxes,
            scores=sample_scores,
        )
        human = result.to_human()

        assert "boxes_xyxy" in human
        assert "boxes_xywh" in human
        assert "boxes_ltwh" in human

    def test_to_dict_is_to_human(
        self,
        sample_boxes: npt.NDArray[np.float64],
    ) -> None:
        """Test to_dict returns same as to_human."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        result = YoloPredictionResult(boxes=sample_boxes)
        assert result.to_dict() == result.to_human()


class TestYoloPredictionResultFiltering:
    def test_filter_by_confidence(
        self,
        sample_image: npt.NDArray[np.uint8],
        sample_class_names: list[str],
    ) -> None:
        """Test filtering by confidence threshold."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        boxes = np.array([[10, 10, 100, 100], [200, 200, 300, 300]])
        scores = np.array([0.9, 0.3])
        result = YoloPredictionResult(
            original_image=sample_image,
            boxes=boxes,
            scores=scores,
            class_names=sample_class_names,
        )

        filtered = result.filter_by_confidence(0.5)
        assert filtered.num_predictions == 1
        assert filtered.scores is not None
        assert filtered.scores[0] == 0.9

    def test_filter_by_class(
        self,
        sample_image: npt.NDArray[np.uint8],
    ) -> None:
        """Test filtering by class name."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        boxes = np.array([[10, 10, 100, 100], [200, 200, 300, 300]])
        scores = np.array([0.9, 0.8])
        class_names = ["cow", "person"]
        result = YoloPredictionResult(
            original_image=sample_image,
            boxes=boxes,
            scores=scores,
            class_names=class_names,
        )

        filtered = result.filter_by_class(["cow"])
        assert filtered.num_predictions == 1
        assert filtered.class_names == ["cow"]

    def test_filter_empty_returns_self(self) -> None:
        """Test filtering empty result returns self."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        result = YoloPredictionResult()
        assert result.filter_by_confidence(0.5) is result
        assert result.filter_by_class(["cow"]) is result


class TestYoloPredictionResultCrops:
    def test_get_crops(
        self,
        sample_image: npt.NDArray[np.uint8],
        sample_boxes: npt.NDArray[np.float64],
    ) -> None:
        """Test extracting crops from detections."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        result = YoloPredictionResult(
            original_image=sample_image,
            boxes=sample_boxes,
        )
        crops = result.get_crops()

        assert len(crops) == 2
        # First box: [100, 50, 300, 250] -> crop shape (200, 200, 3)
        assert crops[0].shape == (200, 200, 3)

    def test_crops_no_image_returns_empty(
        self,
        sample_boxes: npt.NDArray[np.float64],
    ) -> None:
        """Test get_crops without original image returns empty list."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        result = YoloPredictionResult(boxes=sample_boxes)
        assert result.get_crops() == []


class TestYoloPredictionResultVisualization:
    def test_draw_on_image(
        self,
        sample_image: npt.NDArray[np.uint8],
        sample_boxes: npt.NDArray[np.float64],
        sample_scores: npt.NDArray[np.float64],
        sample_class_names: list[str],
    ) -> None:
        """Test draw_on_image returns annotated image."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        result = YoloPredictionResult(
            original_image=sample_image,
            boxes=sample_boxes,
            scores=sample_scores,
            class_names=sample_class_names,
        )
        annotated = result.draw_on_image()

        assert annotated.shape == sample_image.shape
        assert annotated.dtype == np.uint8

    def test_draw_on_external_image(
        self,
        sample_image: npt.NDArray[np.uint8],
        sample_boxes: npt.NDArray[np.float64],
    ) -> None:
        """Test drawing on a different image."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        result = YoloPredictionResult(boxes=sample_boxes)
        annotated = result.draw_on_image(image=sample_image)
        assert annotated.shape == sample_image.shape

    def test_draw_no_image_raises(self) -> None:
        """Test draw_on_image raises when no image available."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        result = YoloPredictionResult(
            boxes=np.array([[10, 10, 100, 100]])
        )
        with pytest.raises(ValueError, match="No image"):
            result.draw_on_image()

    def test_image_height_no_image_raises(self) -> None:
        """Test image_height raises when no image."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        result = YoloPredictionResult()
        with pytest.raises(RuntimeError, match="not an np.ndarray"):
            _ = result.image_height

    def test_image_width_no_image_raises(self) -> None:
        """Test image_width raises when no image."""
        from bovi_yolo.predictors.results import YoloPredictionResult

        result = YoloPredictionResult()
        with pytest.raises(RuntimeError, match="not an np.ndarray"):
            _ = result.image_width
