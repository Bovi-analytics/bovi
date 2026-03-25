"""Tests for format conversion utilities."""

import numpy as np
import pytest

from bovi_core.ml.utils.format_conversion import (
    convert_box_format,
    convert_color_format,
    standardize_base_output,
)


class TestConvertBoxFormat:
    """Test suite for box format conversion."""

    def test_xyxy_to_xywh(self):
        """Test conversion from xyxy to xywh format."""
        boxes_xyxy = np.array([[10, 20, 100, 200]])
        boxes_xywh = convert_box_format(boxes_xyxy, "xyxy", "xywh")

        expected = np.array([[55, 110, 90, 180]])  # center_x, center_y, width, height
        assert np.allclose(boxes_xywh, expected)

    def test_xyxy_to_ltwh(self):
        """Test conversion from xyxy to ltwh format."""
        boxes_xyxy = np.array([[10, 20, 100, 200]])
        boxes_ltwh = convert_box_format(boxes_xyxy, "xyxy", "ltwh")

        expected = np.array([[10, 20, 90, 180]])  # left, top, width, height
        assert np.allclose(boxes_ltwh, expected)

    def test_xywh_to_xyxy(self):
        """Test conversion from xywh to xyxy format."""
        boxes_xywh = np.array([[55, 110, 90, 180]])  # center_x, center_y, w, h
        boxes_xyxy = convert_box_format(boxes_xywh, "xywh", "xyxy")

        expected = np.array([[10, 20, 100, 200]])  # x1, y1, x2, y2
        assert np.allclose(boxes_xyxy, expected)

    def test_xywh_to_ltwh(self):
        """Test conversion from xywh to ltwh format."""
        boxes_xywh = np.array([[55, 110, 90, 180]])
        boxes_ltwh = convert_box_format(boxes_xywh, "xywh", "ltwh")

        expected = np.array([[10, 20, 90, 180]])  # left, top, width, height
        assert np.allclose(boxes_ltwh, expected)

    def test_ltwh_to_xyxy(self):
        """Test conversion from ltwh to xyxy format."""
        boxes_ltwh = np.array([[10, 20, 90, 180]])  # left, top, width, height
        boxes_xyxy = convert_box_format(boxes_ltwh, "ltwh", "xyxy")

        expected = np.array([[10, 20, 100, 200]])  # x1, y1, x2, y2
        assert np.allclose(boxes_xyxy, expected)

    def test_ltwh_to_xywh(self):
        """Test conversion from ltwh to xywh format."""
        boxes_ltwh = np.array([[10, 20, 90, 180]])  # left, top, width, height
        boxes_xywh = convert_box_format(boxes_ltwh, "ltwh", "xywh")

        # left=10, width=90 -> center_x = 10 + 90/2 = 55
        # top=20, height=180 -> center_y = 20 + 180/2 = 110
        # width and height stay the same
        expected = np.array([[55, 110, 90, 180]])  # center_x, center_y, w, h
        assert np.allclose(boxes_xywh, expected)

    def test_same_format_returns_copy(self):
        """Test that converting to same format returns a copy."""
        boxes = np.array([[10, 20, 100, 200]])
        result = convert_box_format(boxes, "xyxy", "xyxy")

        assert np.array_equal(result, boxes)
        assert result is not boxes  # Should be a copy

    def test_empty_boxes(self):
        """Test conversion with empty boxes array."""
        boxes = np.array([]).reshape(0, 4)
        result = convert_box_format(boxes, "xyxy", "xywh")

        assert result.shape == (0, 4)

    def test_multiple_boxes(self):
        """Test conversion with multiple boxes."""
        boxes_xyxy = np.array([
            [10, 20, 100, 200],
            [50, 60, 150, 160],
            [30, 40, 120, 140]
        ])
        boxes_xywh = convert_box_format(boxes_xyxy, "xyxy", "xywh")

        assert boxes_xywh.shape == (3, 4)
        # Check first box
        assert np.allclose(boxes_xywh[0], [55, 110, 90, 180])

    def test_invalid_source_format(self):
        """Test error on invalid source format."""
        boxes = np.array([[10, 20, 100, 200]])

        with pytest.raises(ValueError, match="Unsupported source format"):
            convert_box_format(boxes, "invalid", "xyxy")

    def test_invalid_target_format(self):
        """Test error on invalid target format."""
        boxes = np.array([[10, 20, 100, 200]])

        with pytest.raises(ValueError, match="Unsupported target format"):
            convert_box_format(boxes, "xyxy", "invalid")

    def test_roundtrip_conversion(self):
        """Test that roundtrip conversion preserves values."""
        original = np.array([[10, 20, 100, 200], [50, 60, 150, 160]])

        # xyxy -> xywh -> xyxy
        xywh = convert_box_format(original, "xyxy", "xywh")
        roundtrip = convert_box_format(xywh, "xywh", "xyxy")

        assert np.allclose(roundtrip, original)

        # xyxy -> ltwh -> xyxy
        ltwh = convert_box_format(original, "xyxy", "ltwh")
        roundtrip2 = convert_box_format(ltwh, "ltwh", "xyxy")

        assert np.allclose(roundtrip2, original)


class TestConvertColorFormat:
    """Test suite for color format conversion."""

    def test_bgr_to_rgb(self):
        """Test conversion from BGR to RGB."""
        # Create a simple image with known BGR values
        bgr_image = np.array([
            [[255, 0, 0], [0, 255, 0]],  # Blue, Green
            [[0, 0, 255], [128, 128, 128]]  # Red, Gray
        ], dtype=np.uint8)

        rgb_image = convert_color_format(bgr_image, "bgr", "rgb")

        # BGR [255, 0, 0] -> RGB [0, 0, 255] (Blue)
        assert np.array_equal(rgb_image[0, 0], [0, 0, 255])
        # BGR [0, 255, 0] -> RGB [0, 255, 0] (Green - same)
        assert np.array_equal(rgb_image[0, 1], [0, 255, 0])
        # BGR [0, 0, 255] -> RGB [255, 0, 0] (Red)
        assert np.array_equal(rgb_image[1, 0], [255, 0, 0])

    def test_rgb_to_bgr(self):
        """Test conversion from RGB to BGR."""
        rgb_image = np.array([
            [[255, 0, 0], [0, 255, 0]],  # Red, Green
            [[0, 0, 255], [128, 128, 128]]  # Blue, Gray
        ], dtype=np.uint8)

        bgr_image = convert_color_format(rgb_image, "rgb", "bgr")

        # RGB [255, 0, 0] -> BGR [0, 0, 255] (Red)
        assert np.array_equal(bgr_image[0, 0], [0, 0, 255])
        # RGB [0, 255, 0] -> BGR [0, 255, 0] (Green - same)
        assert np.array_equal(bgr_image[0, 1], [0, 255, 0])
        # RGB [0, 0, 255] -> BGR [255, 0, 0] (Blue)
        assert np.array_equal(bgr_image[1, 0], [255, 0, 0])

    def test_same_format_returns_copy(self):
        """Test that converting to same format returns a copy."""
        image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = convert_color_format(image, "rgb", "rgb")

        assert np.array_equal(result, image)
        assert result is not image  # Should be a copy

    def test_case_insensitive(self):
        """Test that format names are case-insensitive."""
        image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)

        # Should not raise error
        result1 = convert_color_format(image, "BGR", "RGB")
        result2 = convert_color_format(image, "bgr", "rgb")
        result3 = convert_color_format(image, "BgR", "RgB")

        assert np.array_equal(result1, result2)
        assert np.array_equal(result2, result3)

    def test_invalid_conversion(self):
        """Test error on invalid color format."""
        image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Unsupported color conversion"):
            convert_color_format(image, "rgb", "hsv")

    def test_roundtrip_conversion(self):
        """Test that roundtrip conversion preserves values."""
        original = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        # RGB -> BGR -> RGB
        bgr = convert_color_format(original, "rgb", "bgr")
        roundtrip = convert_color_format(bgr, "bgr", "rgb")

        assert np.array_equal(roundtrip, original)


class TestStandardizeBaseOutput:
    """Test suite for standardizing base output format."""

    def test_convert_boxes_to_target_format(self):
        """Test converting boxes to different format."""
        base_dict = {
            "boxes_xyxy": [[10, 20, 100, 200]],
            "scores": [0.95],
            "metadata": {
                "model_type": "yolo",
                "box_format": "xyxy",
            }
        }

        result = standardize_base_output(base_dict, target_box_format="xywh")

        # Should have new format
        assert "boxes_xywh" in result
        assert result["boxes_xywh"] == [[55, 110, 90, 180]]

        # Metadata should be updated
        assert result["metadata"]["box_format"] == "xywh"
        assert result["metadata"]["converted_from"] == "xyxy"

    def test_no_conversion_if_same_format(self):
        """Test that no conversion happens if already in target format."""
        base_dict = {
            "boxes_xyxy": [[10, 20, 100, 200]],
            "metadata": {"box_format": "xyxy"}
        }

        result = standardize_base_output(base_dict, target_box_format="xyxy")

        # Should not add converted_from since no conversion happened
        assert "converted_from" not in result["metadata"]

    def test_no_conversion_if_target_none(self):
        """Test that no conversion happens if target is None."""
        base_dict = {
            "boxes_xyxy": [[10, 20, 100, 200]],
            "metadata": {"box_format": "xyxy"}
        }

        result = standardize_base_output(base_dict, target_box_format=None)

        # Should be unchanged
        assert result["boxes_xyxy"] == [[10, 20, 100, 200]]
        assert result["metadata"]["box_format"] == "xyxy"

    def test_missing_metadata(self):
        """Test handling of dict without metadata."""
        base_dict = {
            "boxes_xyxy": [[10, 20, 100, 200]],
        }

        # Should not crash
        result = standardize_base_output(base_dict, target_box_format="xywh")

        # Result should have metadata field
        assert "metadata" in result

    def test_color_format_note(self):
        """Test that color format conversion is noted but not performed."""
        base_dict = {
            "boxes_xyxy": [[10, 20, 100, 200]],
            "metadata": {
                "box_format": "xyxy",
                "color_format": "bgr"
            }
        }

        result = standardize_base_output(
            base_dict,
            target_color_format="rgb"
        )

        # Should note that conversion is needed
        assert result["metadata"]["target_color_format"] == "rgb"
        assert result["metadata"]["color_conversion_needed"] is True
