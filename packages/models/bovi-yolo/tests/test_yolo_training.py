"""YOLO trainer/evaluator contracts and a minimal native CPU lifecycle."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
from bovi_core.ml import (
    AbstractDataLoader,
    EvaluationContext,
    ResolvedCheckpoint,
    TrainingContext,
)
from bovi_core.ml.models.checkpoints import LocalCheckpointResolver
from bovi_yolo import (
    YOLOEvaluationConfig,
    YOLOEvaluator,
    YOLOModel,
    YOLOModelConfig,
    YOLOModelProvider,
    YOLOTrainer,
    YOLOTrainingConfig,
)
from PIL import Image
from pydantic import ValidationError


def training_context(tmp_path: Path) -> TrainingContext:
    return TrainingContext(run_id=uuid4(), output_dir=tmp_path)


def evaluation_context(tmp_path: Path) -> EvaluationContext:
    return EvaluationContext(
        evaluation_id=uuid4(),
        output_dir=tmp_path,
        split="validation",
        model_version="test",
    )


def test_configs_are_typed_immutable_and_load_from_yaml(yolo_config: object) -> None:
    training = YOLOTrainingConfig.from_config(yolo_config)  # type: ignore[arg-type]
    evaluation = YOLOEvaluationConfig.from_config(yolo_config)  # type: ignore[arg-type]

    assert training.dataset_yaml_path.is_absolute()
    assert training.epochs == 1
    assert evaluation.metrics == ("precision", "recall", "map50", "map50_95")
    with pytest.raises(ValidationError, match="frozen"):
        training.epochs = 2  # type: ignore[misc]


def test_trainer_delegates_to_ultralytics_and_bundles_native_checkpoints(
    tmp_path: Path,
) -> None:
    native_model = MagicMock()
    model = YOLOModel(native_model=native_model, config=YOLOModelConfig(framework="pytorch"))
    context = training_context(tmp_path)

    def train(**kwargs: object) -> object:
        save_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
        weights = save_dir / "weights"
        weights.mkdir(parents=True)
        (weights / "last.pt").write_bytes(b"last-native-checkpoint")
        (weights / "best.pt").write_bytes(b"best-native-checkpoint")
        (save_dir / "results.csv").write_text(
            "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
            "0,0.4,0.5,0.6,0.3\n"
            "1,0.5,0.6,0.7,0.4\n",
            encoding="utf-8",
        )
        native_model.trainer = SimpleNamespace(
            save_dir=save_dir,
            last=weights / "last.pt",
            best=weights / "best.pt",
            train_loader=SimpleNamespace(dataset=[object(), object()]),
        )
        return {"metrics/mAP50(B)": 0.7, "metrics/mAP50-95(B)": 0.4}

    native_model.train.side_effect = train
    result = YOLOTrainer(
        model=model,
        dataloaders={},
        config=YOLOTrainingConfig(
            dataset_yaml_path=tmp_path / "dataset.yaml",
            epochs=2,
            batch_size=1,
        ),
        context=context,
    ).train()

    assert result.status == "completed"
    assert result.num_examples == 2
    assert result.num_examples_processed == 4
    assert result.best_epoch == 2
    assert [epoch.metrics["map50_95"] for epoch in result.epochs] == [0.3, 0.4]
    assert result.last_checkpoint is not None
    assert result.best_checkpoint is not None
    native_model.train.assert_called_once()
    assert native_model.train.call_args.kwargs["data"].endswith("dataset.yaml")

    resolved = LocalCheckpointResolver().resolve(result.best_checkpoint)
    assert resolved.local_path is not None
    assert resolved.local_path.read_bytes() == b"best-native-checkpoint"
    with patch("bovi_yolo.models.provider.YOLO") as yolo_cls:
        restored = YOLOModelProvider().restore_checkpoint(
            YOLOModelConfig(framework="pytorch"),
            cast(ResolvedCheckpoint[object], resolved),
        )
    yolo_cls.assert_called_once_with(str(resolved.local_path), task="detect")
    assert restored.native_model is yolo_cls.return_value


def test_evaluator_delegates_to_native_validation(tmp_path: Path) -> None:
    native_model = MagicMock()
    native_model.val.return_value = SimpleNamespace(
        results_dict={
            "metrics/precision(B)": 0.8,
            "metrics/recall(B)": 0.7,
            "metrics/mAP50(B)": 0.75,
            "metrics/mAP50-95(B)": 0.5,
            "fitness": 0.525,
        }
    )
    model = YOLOModel(native_model=native_model, config=YOLOModelConfig(framework="pytorch"))
    evaluator = YOLOEvaluator(
        model,
        YOLOEvaluationConfig(
            dataset_yaml_path=tmp_path / "dataset.yaml",
            metrics=("map50", "map50_95"),
        ),
    )

    result = evaluator.evaluate(
        cast(AbstractDataLoader, MagicMock()),
        evaluation_context(tmp_path),
    )

    assert result.status == "completed"
    assert result.metrics == {"map50": 0.75, "map50_95": 0.5}
    native_model.val.assert_called_once()
    assert native_model.val.call_args.kwargs["data"].endswith("dataset.yaml")


@pytest.mark.integration
def test_minimal_native_cpu_training_evaluation_and_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise real Ultralytics code using a generated model and no downloads."""
    monkeypatch.setenv("ULTRALYTICS_OFFLINE", "true")
    dataset_yaml = _write_detection_dataset(tmp_path)
    model = YOLOModelProvider().create(
        YOLOModelConfig(framework="pytorch", model_source="yolo11n.yaml")
    )
    train_result = YOLOTrainer(
        model,
        {},
        YOLOTrainingConfig(
            dataset_yaml_path=dataset_yaml,
            epochs=1,
            image_size=64,
            batch_size=1,
            device="cpu",
            workers=0,
            reuse_model_weights=True,
        ),
        training_context(tmp_path / "training"),
    ).train()

    assert train_result.status == "completed", train_result.issues
    assert train_result.best_checkpoint is not None
    resolved = LocalCheckpointResolver().resolve(train_result.best_checkpoint)
    restored = YOLOModelProvider().restore_checkpoint(
        YOLOModelConfig(framework="pytorch", model_source="yolo11n.yaml"),
        resolved,
    )
    evaluation = YOLOEvaluator(
        restored,
        YOLOEvaluationConfig(
            dataset_yaml_path=dataset_yaml,
            image_size=64,
            batch_size=1,
            workers=0,
        ),
    ).evaluate(
        cast(AbstractDataLoader, MagicMock()),
        evaluation_context(tmp_path / "evaluation"),
    )
    assert evaluation.status == "completed", evaluation.issues
    assert evaluation.num_examples == 1
    assert "map50_95" in evaluation.metrics


def _write_detection_dataset(root: Path) -> Path:
    for split in ("train", "val"):
        images = root / "dataset" / split / "images"
        labels = root / "dataset" / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        pixels = np.zeros((32, 32, 3), dtype=np.uint8)
        pixels[8:24, 8:24] = 255
        Image.fromarray(pixels).save(images / "cow.jpg")
        (labels / "cow.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    dataset_yaml = root / "dataset.yaml"
    dataset_yaml.write_text(
        f"path: {root / 'dataset'}\ntrain: train/images\nval: val/images\nnames:\n  0: cow\n",
        encoding="utf-8",
    )
    return dataset_yaml
