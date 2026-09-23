"""YOLO-specific prediction result implementation.

Three-level serialization:
- Level 1 (raw): ultralytics.Results object
- Level 2 (base): Minimal dict for MLflow signatures
- Level 3 (rich): Full YoloPredictionResult with all methods
"""

from __future__ import annotations

from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from bovi_core.ml.predictors.results import HumanReadablePredictionResult
from matplotlib.patches import Rectangle


class YoloPredictionResult(HumanReadablePredictionResult):
    """YOLO-specific prediction result with visualization and manipulation.

    Provides a rich interface for working with YOLO detection/segmentation
    results, including:
    - Multiple box format conversions (xyxy, xywh, ltwh)
    - Filtering by confidence and class
    - Image cropping
    - Visualization with OpenCV and Matplotlib
    - Serialization for MLflow and pipelines
    """

    def __init__(
        self,
        original_image: npt.NDArray[np.uint8] | None = None,
        boxes: npt.NDArray[np.float64] | None = None,
        masks: list[npt.NDArray[np.uint8]] | None = None,
        class_ids: npt.NDArray[np.float64] | None = None,
        class_names: list[str] | None = None,
        scores: npt.NDArray[np.float64] | None = None,
        raw_result: object | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize YOLO prediction result.

        Args:
            original_image: Original input image (for visualization).
            boxes: Bounding boxes in xyxy format [x1, y1, x2, y2].
            masks: List of segmentation masks (instance segmentation).
            class_ids: Class ID for each detection.
            class_names: Class name for each detection.
            scores: Confidence score for each detection.
            raw_result: Original YOLO Results object (Level 1).
            metadata: Additional metadata.
        """
        super().__init__(metadata)
        self.original_image = original_image
        self.boxes = boxes
        self.masks = masks
        self.class_ids = class_ids
        self.class_names = class_names
        self.scores = scores
        self.raw_result = raw_result

        if self.boxes is not None and not isinstance(self.boxes, np.ndarray):
            self.boxes = np.array(self.boxes)

    @classmethod
    def from_raw(
        cls,
        raw_output: Any,
        **kwargs: Any,
    ) -> YoloPredictionResult:
        """Create YoloPredictionResult from YOLO model output (Level 1 -> Level 3).

        Args:
            raw_output: YOLO model prediction results (ultralytics.Results list).
            **kwargs: Additional context:
                original_image: Original input image (extracted from results if None).
                class_names_map: Optional mapping of class IDs to names.

        Returns:
            YoloPredictionResult instance containing the YOLO predictions.
        """
        original_image: npt.NDArray[np.uint8] | None = kwargs.get("original_image", None)
        class_names_map: dict[int, str] | None = kwargs.get("class_names_map", None)

        if not raw_output:
            return cls()

        result = raw_output[0]
        original_image = original_image if original_image is not None else result.orig_img

        boxes = None
        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()

        masks = None
        if hasattr(result, "masks") and result.masks is not None:
            masks = result.masks.data.cpu().numpy()

        class_ids = None
        class_names = None
        if hasattr(result, "boxes") and result.boxes is not None:
            class_ids = result.boxes.cls.cpu().numpy()
            if class_names_map is None and hasattr(result, "names"):
                class_names_map = result.names
            if class_names_map is not None:
                class_names = [class_names_map[int(i)] for i in class_ids]

        scores = None
        if hasattr(result, "boxes") and result.boxes is not None:
            scores = result.boxes.conf.cpu().numpy()

        return cls(
            original_image=original_image,
            boxes=boxes,
            masks=masks,
            class_ids=class_ids,
            class_names=class_names,
            scores=scores,
            raw_result=None,
            metadata={"model_type": "yolo"},
        )

    def to_serializable(self) -> dict[str, Any]:
        """Convert to base-level dict (Level 2) for MLflow signatures.

        Returns minimal, portable dictionary with boxes in xyxy format,
        format metadata, no images, and JSON-serializable values.

        Returns:
            Serializable dict ready for MLflow or model chaining.
        """
        return {
            "boxes_xyxy": self.boxes.tolist()
            if self.boxes is not None and len(self.boxes) > 0
            else None,
            "class_ids": self.class_ids.tolist()
            if self.class_ids is not None and len(self.class_ids) > 0
            else None,
            "class_names": self.class_names,
            "scores": self.scores.tolist()
            if self.scores is not None and len(self.scores) > 0
            else None,
            "num_predictions": self.num_predictions,
            "has_masks": self.masks is not None,
            "metadata": {
                **self.metadata,
                "model_type": "yolo",
                "box_format": "xyxy",
                "color_format": "bgr",
                "image_shape": list(self.original_image.shape)
                if self.original_image is not None
                else None,
            },
        }

    def to_human(self) -> dict[str, Any]:
        """Rich dict with all formats and computed fields.

        Returns comprehensive dictionary with all data from to_serializable()
        plus multiple box formats (xyxy, xywh, ltwh).

        Returns:
            Rich dict with all available information.
        """
        return {
            **self.to_serializable(),
            "boxes_xywh": self.get_boxes_xywh().tolist()
            if self.boxes is not None and len(self.boxes) > 0
            else None,
            "boxes_ltwh": self.get_boxes_ltwh().tolist()
            if self.boxes is not None and len(self.boxes) > 0
            else None,
        }

    @property
    def num_predictions(self) -> int:
        """Return the number of detected objects."""
        if self.boxes is not None:
            return len(self.boxes)
        elif self.masks is not None:
            return len(self.masks)
        return 0

    @property
    def image_height(self) -> int:
        """Return the height of the original image."""
        if type(self.original_image) is np.ndarray:
            return int(self.original_image.shape[0])
        raise RuntimeError("Image is not an np.ndarray. Could not get height.")

    @property
    def image_width(self) -> int:
        """Return the width of the original image."""
        if type(self.original_image) is np.ndarray:
            return int(self.original_image.shape[1])
        raise RuntimeError("Image is not an np.ndarray. Could not get width.")

    def get_boxes_xyxy(self) -> npt.NDArray[np.float64]:
        """Return boxes in xyxy format (x1, y1, x2, y2)."""
        return self.boxes if self.boxes is not None else np.array([])

    def get_boxes_xywh(self) -> npt.NDArray[np.float64]:
        """Convert and return boxes in xywh format (center_x, center_y, w, h)."""
        if self.boxes is None or len(self.boxes) == 0:
            return np.array([])

        boxes = self.boxes.copy()
        result = np.zeros_like(boxes)
        result[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
        result[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
        result[:, 2] = boxes[:, 2] - boxes[:, 0]
        result[:, 3] = boxes[:, 3] - boxes[:, 1]
        return result

    def get_boxes_ltwh(self) -> npt.NDArray[np.float64]:
        """Return boxes in ltwh format (left, top, width, height)."""
        if self.boxes is None or len(self.boxes) == 0:
            return np.array([])

        boxes = self.boxes.copy()
        result = np.zeros_like(boxes)
        result[:, 0] = boxes[:, 0]
        result[:, 1] = boxes[:, 1]
        result[:, 2] = boxes[:, 2] - boxes[:, 0]
        result[:, 3] = boxes[:, 3] - boxes[:, 1]
        return result

    def get_crops(self) -> list[npt.NDArray[np.uint8]]:
        """Extract image crops based on bounding boxes.

        Returns:
            List of cropped image arrays, one per detection.
        """
        if self.boxes is None or len(self.boxes) == 0 or self.original_image is None:
            return []

        crops: list[npt.NDArray[np.uint8]] = []
        for box in self.boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = self.original_image[y1:y2, x1:x2].copy()
            crops.append(crop)
        return crops

    def filter_by_confidence(self, threshold: float) -> YoloPredictionResult:
        """Filter detections by confidence score.

        Args:
            threshold: Minimum confidence score to keep.

        Returns:
            New YoloPredictionResult with only high-confidence detections.
        """
        if self.scores is None or len(self.scores) == 0:
            return self

        mask = self.scores >= threshold
        return self._filter_by_mask(mask)

    def filter_by_class(self, class_names: list[str]) -> YoloPredictionResult:
        """Filter detections by class names.

        Args:
            class_names: List of class names to keep.

        Returns:
            New YoloPredictionResult with only the specified classes.
        """
        if self.class_names is None or len(self.class_names) == 0:
            return self

        mask = np.array([name in class_names for name in self.class_names])
        return self._filter_by_mask(mask)

    def _filter_by_mask(self, mask: npt.NDArray[np.bool_]) -> YoloPredictionResult:
        """Filter results by a boolean mask.

        Args:
            mask: Boolean array indicating which detections to keep.

        Returns:
            New YoloPredictionResult with only the filtered detections.
        """
        return YoloPredictionResult(
            original_image=self.original_image,
            boxes=self.boxes[mask] if self.boxes is not None and len(self.boxes) > 0 else None,
            masks=[self.masks[i] for i, m in enumerate(mask) if m]
            if self.masks is not None
            else None,
            class_ids=self.class_ids[mask]
            if self.class_ids is not None and len(self.class_ids) > 0
            else None,
            class_names=[self.class_names[i] for i, m in enumerate(mask) if m]
            if self.class_names is not None
            else None,
            scores=self.scores[mask] if self.scores is not None and len(self.scores) > 0 else None,
            raw_result=self.raw_result,
            metadata=self.metadata,
        )

    def draw_on_image(
        self,
        image: npt.NDArray[np.uint8] | None = None,
        draw_boxes: bool = True,
        draw_masks: bool = True,
        draw_labels: bool = True,
        colors: list[tuple[int, int, int]] | None = None,
        box_thickness: int = 2,
        mask_alpha: float = 0.5,
    ) -> npt.NDArray[np.uint8]:
        """Draw detection results on an image.

        Args:
            image: Image to draw on. If None, uses the original image.
            draw_boxes: Whether to draw bounding boxes.
            draw_masks: Whether to draw segmentation masks.
            draw_labels: Whether to draw class labels and scores.
            colors: List of BGR colors per detection. Auto-generated if None.
            box_thickness: Thickness of bounding box lines.
            mask_alpha: Transparency of the masks.

        Returns:
            Image with detections drawn on it.

        Raises:
            ValueError: If no image is available.
        """
        if image is None:
            if self.original_image is None:
                raise ValueError("No image provided and original_image is None")
            image = self.original_image.copy()

        assert isinstance(image, np.ndarray)

        if colors is None:
            np.random.seed(42)
            colors = [
                tuple(np.random.randint(0, 255, 3).tolist())
                for _ in range(max(1, self.num_predictions))
            ]

        if draw_masks and self.masks is not None:
            for i, mask in enumerate(self.masks):
                color = colors[i]
                colored_mask = np.zeros_like(image, dtype=np.uint8)
                mask_3d = np.stack([mask] * 3, axis=2) if mask.ndim == 2 else mask
                colored_mask[mask_3d > 0] = color
                image = np.asarray(
                    cv2.addWeighted(image, 1, colored_mask, mask_alpha, 0),
                    dtype=np.uint8,
                )

        if draw_boxes and self.boxes is not None and len(self.boxes) > 0:
            for i, box in enumerate(self.boxes):
                x1, y1, x2, y2 = map(int, box)
                color = colors[i]
                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    color,
                    box_thickness,  # type: ignore[arg-type]
                )

                if draw_labels:
                    label = self._build_label(i)
                    if label:
                        self._draw_label(image, label, x1, y1, color)

        return image

    def _build_label(self, index: int) -> str:
        """Build label string for a detection at given index.

        Args:
            index: Detection index.

        Returns:
            Formatted label string.
        """
        label = ""
        if self.class_names is not None and index < len(self.class_names):
            label += self.class_names[index]
        if self.scores is not None and index < len(self.scores):
            label += f": {self.scores[index]:.2f}" if label else f"{self.scores[index]:.2f}"
        return label

    def _draw_label(
        self,
        image: npt.NDArray[np.uint8],
        label: str,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
    ) -> None:
        """Draw a label with background on an image.

        Args:
            image: Image to draw on.
            label: Text to draw.
            x1: Left x coordinate.
            y1: Top y coordinate.
            color: BGR color for background.
        """
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(
            image,
            (x1, y1 - text_size[1] - 5),
            (x1 + text_size[0], y1),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    def visualize(self, **kwargs: Any) -> None:
        """Visualize the detections using matplotlib.

        Args:
            **kwargs: Visualization options:
                figsize: Figure size as (width, height) tuple. Default (10, 10).
                alpha: Transparency of the masks. Default 0.5.

        Raises:
            ValueError: If original_image is None.
        """
        figsize: tuple[int, int] = kwargs.get("figsize", (10, 10))
        alpha: float = kwargs.get("alpha", 0.5)

        if self.original_image is None:
            raise ValueError("Cannot visualize: original_image is None")
        plt.figure(figsize=figsize)
        plt.imshow(self.original_image)

        colors = plt.colormaps["tab10"](np.linspace(0, 1, 10))

        if self.boxes is not None:
            for i, box in enumerate(self.boxes):
                x1, y1, x2, y2 = box
                width, height = x2 - x1, y2 - y1

                rect = Rectangle(
                    (x1, y1),
                    width,
                    height,
                    fill=False,
                    edgecolor=colors[i % 10],
                    linewidth=2,
                )
                plt.gca().add_patch(rect)

                if self.class_names is not None and self.scores is not None:
                    label = f"{self.class_names[i]}: {self.scores[i]:.2f}"
                    plt.text(
                        x1,
                        y1 - 5,
                        label,
                        fontsize=10,
                        bbox=dict(facecolor=colors[i % 10], alpha=0.5),
                    )

        if self.masks is not None:
            for i, mask in enumerate(self.masks):
                color_mask = np.zeros((mask.shape[0], mask.shape[1], 4))
                color = colors[i % 10]
                color_mask[mask > 0] = (*color[:3], alpha)
                plt.imshow(color_mask)

        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def to_dict(self) -> dict[str, Any]:
        """Legacy method for backward compatibility.

        New code should use to_serializable() or to_human() instead.

        Returns:
            Rich dict with all formats (equivalent to to_human()).
        """
        return self.to_human()
