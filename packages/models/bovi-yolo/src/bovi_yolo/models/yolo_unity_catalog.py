"""Unity Catalog wrapper for YOLO detection model.

Extends PyTorchModelWrapper with ultralytics-specific model loading.
"""

from __future__ import annotations

from typing import Any

from bovi_core.ml.models.pytorch_model_wrapper import PyTorchModelWrapper
from typing_extensions import override


class YOLOModelWrapper(PyTorchModelWrapper):
    """Pyfunc wrapper for YOLO detection model.

    Overrides load_context to use ultralytics YOLO loading
    instead of standard torch.load().
    """

    @override
    def load_context(self, context: Any) -> None:
        """Load YOLO model from artifacts via ultralytics.

        Args:
            context: MLflow PythonModelContext with artifacts.
        """
        from ultralytics import YOLO

        model_path = context.artifacts["model_path"]
        self.model = YOLO(model_path)

    @override
    def predict(
        self,
        context: Any,
        model_input: Any,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Run YOLO inference (used during log_model validation).

        Args:
            context: MLflow context.
            model_input: Input image data.
            params: Optional parameters.

        Returns:
            YOLO prediction results.
        """
        return self.model(model_input)
