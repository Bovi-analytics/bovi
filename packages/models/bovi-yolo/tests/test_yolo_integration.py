"""Integration tests for YOLO pipeline.

Tests end-to-end: source -> dataset -> model -> predict.
Model-dependent tests are skipped if weights are not available.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from bovi_core.ml.dataloaders.sources import LocalFileSource

if TYPE_CHECKING:
    from bovi_core.config import Config

WEIGHTS_PATH = Path("data/experiments/yolo/versions/v1/weights/yolo12n.pt")


def _weights_available() -> bool:
    """Check if YOLO weights are available."""
    return WEIGHTS_PATH.exists()


class TestSourceToDatasetPipeline:
    def test_source_to_dataset(self, temp_image_dir: Path) -> None:
        """Test creating dataset from source."""
        from bovi_yolo.dataloaders.datasets import YOLODataset

        source = LocalFileSource(
            root_dir=temp_image_dir / "train" / "images",
            file_pattern="*.jpeg",
        )
        dataset = YOLODataset(source=source)

        assert len(dataset) == 1
        item = dataset[0]
        assert "image" in item
        assert "metadata" in item
        assert isinstance(item["image"], np.ndarray)
        assert item["image"].dtype == np.uint8

    def test_multiple_splits(self, temp_image_dir: Path) -> None:
        """Test loading from multiple data splits."""
        from bovi_yolo.dataloaders.datasets import YOLODataset

        for split, pattern in [
            ("train", "*.jpeg"),
            ("val", "*.jpg"),
            ("test", "*.jpg"),
        ]:
            source = LocalFileSource(
                root_dir=temp_image_dir / split / "images",
                file_pattern=pattern,
            )
            dataset = YOLODataset(source=source)
            assert len(dataset) == 1, f"Expected 1 image for {split}"


class TestTransformPipeline:
    def test_validation_on_dataset_output(self, temp_image_dir: Path) -> None:
        """Test ImageValidationTransform works on dataset output."""
        from bovi_yolo.dataloaders.datasets import YOLODataset
        from bovi_yolo.dataloaders.transforms import (
            ImageValidationTransform,
        )

        source = LocalFileSource(
            root_dir=temp_image_dir / "train" / "images",
            file_pattern="*.jpeg",
        )
        dataset = YOLODataset(source=source)
        item = dataset[0]

        transform = ImageValidationTransform()
        validated = transform(**item)
        assert np.array_equal(validated["image"], item["image"])

    def test_resize_on_dataset_output(self, temp_image_dir: Path) -> None:
        """Test ImageResizeTransform works on dataset output."""
        from bovi_yolo.dataloaders.datasets import YOLODataset
        from bovi_yolo.dataloaders.transforms import ImageResizeTransform

        source = LocalFileSource(
            root_dir=temp_image_dir / "train" / "images",
            file_pattern="*.jpeg",
        )
        dataset = YOLODataset(source=source)
        item = dataset[0]

        transform = ImageResizeTransform(target_size=(320, 320))
        resized = transform(**item)
        assert resized["image"].shape[:2] == (320, 320)


class TestConfigDrivenPipeline:
    def test_source_from_config(self, yolo_config: Config) -> None:
        """Test creating source from config."""
        from bovi_yolo.dataloaders.sources import YOLOImageSource

        source = YOLOImageSource.from_config(yolo_config, split="inference")
        assert len(source) >= 1

    def test_transforms_from_config(self, yolo_config: Config) -> None:
        """Test building transforms from config."""
        from bovi_core.ml.dataloaders.transforms.registry import (
            TransformRegistry,
        )

        transforms = TransformRegistry.from_config(
            yolo_config.experiment.models.yolo.dataloaders.inference.transforms
        )
        assert len(transforms) >= 1
        assert "image_validation" in transforms

    def test_configured_transforms_run_in_vision_pipeline(
        self,
        yolo_config: Config,
        sample_image: np.ndarray,
    ) -> None:
        """YOLO transforms compose through the loader's Albumentations contract."""
        from bovi_core.ml.dataloaders.transforms import build_vision_pipeline

        pipeline = build_vision_pipeline(
            yolo_config.experiment.models.yolo.dataloaders.inference.transforms
        )

        result = pipeline(image=sample_image)

        assert result["image"].shape == (640, 640, 3)
        assert result["image"].dtype == sample_image.dtype


@pytest.mark.skipif(
    not _weights_available(),
    reason="YOLO weights not available",
)
class TestEndToEndPipeline:
    def test_full_pipeline(self, yolo_config: Config, temp_image_dir: Path) -> None:
        """Test full pipeline: source -> dataset -> model -> predict."""
        from bovi_core.ml import ResolvedModelArtifact
        from bovi_yolo.dataloaders.datasets import YOLODataset
        from bovi_yolo.models import YOLOModelConfig, YOLOModelProvider
        from bovi_yolo.predictors import YOLOPredictor
        from bovi_yolo.predictors.results import YoloPredictionResult

        # Source + Dataset
        source = LocalFileSource(
            root_dir=temp_image_dir / "train" / "images",
            file_pattern="*.jpeg",
        )
        dataset = YOLODataset(source=source)

        model = YOLOModelProvider().load_artifact(
            YOLOModelConfig.from_config(yolo_config),
            ResolvedModelArtifact[object](
                format="ultralytics-pt",
                source_uri=WEIGHTS_PATH.resolve().as_uri(),
                local_path=WEIGHTS_PATH,
            ),
        )
        predictor = YOLOPredictor(model=model, config=yolo_config)

        # Predict
        image = dataset[0]["image"]
        result = predictor.predict(image, return_format="rich")

        assert isinstance(result, YoloPredictionResult)
        assert result.num_predictions >= 0

    def test_three_level_returns(self, yolo_config: Config, temp_image_dir: Path) -> None:
        """Test all three return formats work."""
        from bovi_core.ml import ResolvedModelArtifact
        from bovi_yolo.dataloaders.datasets import YOLODataset
        from bovi_yolo.models import YOLOModelConfig, YOLOModelProvider
        from bovi_yolo.predictors import YOLOPredictor
        from bovi_yolo.predictors.results import YoloPredictionResult

        source = LocalFileSource(
            root_dir=temp_image_dir / "train" / "images",
            file_pattern="*.jpeg",
        )
        dataset = YOLODataset(source=source)
        image = dataset[0]["image"]

        model = YOLOModelProvider().load_artifact(
            YOLOModelConfig.from_config(yolo_config),
            ResolvedModelArtifact[object](
                format="ultralytics-pt",
                source_uri=WEIGHTS_PATH.resolve().as_uri(),
                local_path=WEIGHTS_PATH,
            ),
        )
        predictor = YOLOPredictor(model=model, config=yolo_config)

        raw = predictor.predict(image, return_format="raw")
        assert raw is not None

        base = predictor.predict(image, return_format="base")
        assert isinstance(base, dict)

        rich = predictor.predict(image, return_format="rich")
        assert isinstance(rich, YoloPredictionResult)
