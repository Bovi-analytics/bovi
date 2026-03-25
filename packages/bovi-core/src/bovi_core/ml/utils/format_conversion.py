"""Utility functions for format conversion between different box formats and color spaces.

These utilities allow downstream models to convert prediction outputs to their
expected formats based on metadata.
"""

from typing import Any, Dict, Optional

import cv2
import numpy as np


def convert_box_format(
    boxes: np.ndarray, from_format: str, to_format: str
) -> np.ndarray:
    """
    Convert bounding boxes between different formats.

    Supported formats:
    - "xyxy": [x1, y1, x2, y2] - top-left and bottom-right corners
    - "xywh": [center_x, center_y, width, height] - center point and dimensions
    - "ltwh": [left, top, width, height] - top-left corner and dimensions

    Args:
        boxes: Bounding boxes array of shape (N, 4)
        from_format: Source format ("xyxy", "xywh", or "ltwh")
        to_format: Target format ("xyxy", "xywh", or "ltwh")

    Returns:
        Converted boxes array of shape (N, 4)

    Raises:
        ValueError: If format is not supported

    Example:
        >>> boxes_xyxy = np.array([[10, 20, 100, 200]])
        >>> boxes_xywh = convert_box_format(boxes_xyxy, "xyxy", "xywh")
        >>> print(boxes_xywh)
        [[55. 110. 90. 180.]]  # center_x, center_y, width, height
    """
    if from_format == to_format:
        return boxes.copy()

    if boxes.size == 0:
        return boxes.copy()

    boxes = boxes.copy()
    result = np.zeros_like(boxes)

    # Convert from source format to xyxy (intermediate)
    if from_format == "xyxy":
        xyxy = boxes
    elif from_format == "xywh":
        # xywh -> xyxy
        result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = center_x - width/2
        result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = center_y - height/2
        result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 = center_x + width/2
        result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 = center_y + height/2
        xyxy = result
    elif from_format == "ltwh":
        # ltwh -> xyxy
        result[:, 0] = boxes[:, 0]  # x1 = left
        result[:, 1] = boxes[:, 1]  # y1 = top
        result[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = left + width
        result[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = top + height
        xyxy = result
    else:
        raise ValueError(
            f"Unsupported source format: {from_format}. "
            f"Must be one of: 'xyxy', 'xywh', 'ltwh'"
        )

    # Convert from xyxy to target format
    if to_format == "xyxy":
        return xyxy
    elif to_format == "xywh":
        # xyxy -> xywh (need fresh array for output)
        output = np.zeros_like(xyxy)
        output[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2  # center_x
        output[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2  # center_y
        output[:, 2] = xyxy[:, 2] - xyxy[:, 0]  # width
        output[:, 3] = xyxy[:, 3] - xyxy[:, 1]  # height
        return output
    elif to_format == "ltwh":
        # xyxy -> ltwh (need fresh array for output)
        output = np.zeros_like(xyxy)
        output[:, 0] = xyxy[:, 0]  # left
        output[:, 1] = xyxy[:, 1]  # top
        output[:, 2] = xyxy[:, 2] - xyxy[:, 0]  # width
        output[:, 3] = xyxy[:, 3] - xyxy[:, 1]  # height
        return output
    else:
        raise ValueError(
            f"Unsupported target format: {to_format}. "
            f"Must be one of: 'xyxy', 'xywh', 'ltwh'"
        )


def convert_color_format(
    image: np.ndarray, from_format: str, to_format: str
) -> np.ndarray:
    """
    Convert image color format between RGB and BGR.

    Args:
        image: Image array of shape (H, W, 3)
        from_format: Source format ("rgb" or "bgr")
        to_format: Target format ("rgb" or "bgr")

    Returns:
        Converted image array

    Raises:
        ValueError: If format is not supported

    Example:
        >>> bgr_image = cv2.imread("image.jpg")  # OpenCV loads as BGR
        >>> rgb_image = convert_color_format(bgr_image, "bgr", "rgb")
    """
    if from_format == to_format:
        return image.copy()

    from_format = from_format.lower()
    to_format = to_format.lower()

    if from_format == "bgr" and to_format == "rgb":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif from_format == "rgb" and to_format == "bgr":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(
            f"Unsupported color conversion: {from_format} -> {to_format}. "
            f"Formats must be 'rgb' or 'bgr'"
        )


def standardize_base_output(
    base_dict: Dict[str, Any],
    target_box_format: Optional[str] = None,
    target_color_format: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert base-level prediction dict to target formats based on metadata.

    This function is useful when passing predictions to downstream models that
    expect specific formats. It uses the metadata to determine the current format
    and converts to the target format if needed.

    Args:
        base_dict: Base-level prediction dict with metadata (from to_serializable())
        target_box_format: Desired box format ("xyxy", "xywh", "ltwh", or None to keep original)
        target_color_format: Desired color format ("rgb", "bgr", or None to keep original)

    Returns:
        Converted base dict with updated metadata

    Example:
        >>> # YOLO outputs boxes in xyxy format
        >>> yolo_output = yolo_result.to_serializable()
        >>> print(yolo_output["metadata"]["box_format"])
        "xyxy"

        >>> # Convert to xywh for a model that expects center coordinates
        >>> converted = standardize_base_output(yolo_output, target_box_format="xywh")
        >>> print(converted["metadata"]["box_format"])
        "xywh"
        >>> print(converted["boxes_xywh"])  # New key with converted boxes
    """
    metadata = base_dict.get("metadata", {})
    current_box_format = metadata.get("box_format")
    current_color_format = metadata.get("color_format")

    result = base_dict.copy()
    result["metadata"] = metadata.copy()

    # Convert boxes if needed
    if (
        target_box_format
        and current_box_format
        and target_box_format != current_box_format
    ):
        # Find the box key (could be boxes_xyxy, boxes_xywh, etc.)
        box_key = f"boxes_{current_box_format}"
        if box_key in base_dict and base_dict[box_key] is not None:
            boxes = np.array(base_dict[box_key])
            converted = convert_box_format(
                boxes, current_box_format, target_box_format
            )
            # Add new key with target format
            result[f"boxes_{target_box_format}"] = converted.tolist()
            # Update metadata
            result["metadata"]["box_format"] = target_box_format
            result["metadata"]["converted_from"] = current_box_format

    # Note: Color conversion would require image data, which is not included
    # in base-level serialization (by design - base level is lightweight)
    # If color conversion is needed, it should be done at the rich level before
    # serialization, or the image should be passed separately

    if target_color_format and current_color_format:
        if target_color_format != current_color_format:
            # Just update metadata - actual conversion requires image
            result["metadata"]["target_color_format"] = target_color_format
            result["metadata"]["color_conversion_needed"] = True

    return result
