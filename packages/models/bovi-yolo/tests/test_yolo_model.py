"""Tests for the YOLO runtime model and provider."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from bovi_core.ml import (
    ModelProviderRegistry,
    ResolvedCheckpoint,
    ResolvedModelArtifact,
)


def test_model_wraps_and_calls_native_model() -> None:
    from bovi_yolo.models import YOLOModel, YOLOModelConfig

    native_model = MagicMock(return_value=["result"])
    config = YOLOModelConfig(framework="pytorch")
    model = YOLOModel(native_model=native_model, config=config)

    assert model.native_model is native_model
    assert model.config is config
    assert model("image", conf=0.5) == ["result"]
    native_model.assert_called_once_with("image", conf=0.5)


def test_model_config_uses_defaults_for_existing_config(yolo_config: object) -> None:
    from bovi_yolo.models import YOLOModelConfig

    config = YOLOModelConfig.from_config(yolo_config)  # type: ignore[arg-type]

    assert config.framework == "pytorch"
    assert config.model_source == "yolo12n.yaml"
    assert config.task == "detect"


class TestYOLOModelProvider:
    @patch("bovi_yolo.models.yolo_provider.YOLO")
    def test_create_builds_fresh_native_model(self, yolo_cls: MagicMock) -> None:
        from bovi_yolo.models import YOLOModelConfig, YOLOModelProvider

        native_model = MagicMock()
        yolo_cls.return_value = native_model
        config = YOLOModelConfig(
            framework="pytorch",
            model_source="custom.yaml",
            task="detect",
        )

        model = YOLOModelProvider().create(config)

        yolo_cls.assert_called_once_with("custom.yaml", task="detect")
        assert model.native_model is native_model
        assert model.config is config

    @patch("bovi_yolo.models.yolo_provider.YOLO")
    def test_restore_checkpoint_loads_local_path(
        self,
        yolo_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        from bovi_yolo.models import YOLOModelConfig, YOLOModelProvider

        checkpoint_path = tmp_path / "last.pt"
        checkpoint_path.touch()
        checkpoint = ResolvedCheckpoint[object](
            format="ultralytics-pt",
            source_uri=checkpoint_path.as_uri(),
            local_path=checkpoint_path,
        )
        config = YOLOModelConfig(framework="pytorch")

        model = YOLOModelProvider().restore_checkpoint(config, checkpoint)

        yolo_cls.assert_called_once_with(str(checkpoint_path), task="detect")
        assert model.native_model is yolo_cls.return_value

    @patch("bovi_yolo.models.yolo_provider.YOLO")
    def test_load_artifact_loads_local_path(
        self,
        yolo_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        from bovi_yolo.models import YOLOModelConfig, YOLOModelProvider

        artifact_path = tmp_path / "best.pt"
        artifact_path.touch()
        artifact = ResolvedModelArtifact[object](
            format="ultralytics-pt",
            source_uri=artifact_path.as_uri(),
            local_path=artifact_path,
        )
        config = YOLOModelConfig(framework="pytorch")

        model = YOLOModelProvider().load_artifact(config, artifact)

        yolo_cls.assert_called_once_with(str(artifact_path), task="detect")
        assert model.native_model is yolo_cls.return_value

    @pytest.mark.parametrize("resource_format", ["ultralytics-pt", "ultralytics-runtime"])
    @patch("bovi_yolo.models.yolo_provider.YOLO")
    def test_resolved_native_payload_avoids_reloading(
        self,
        yolo_cls: MagicMock,
        resource_format: str,
    ) -> None:
        from bovi_yolo.models import YOLOModelConfig, YOLOModelProvider

        native_model = yolo_cls.return_value
        checkpoint = ResolvedCheckpoint[object](
            format=resource_format,
            source_uri="memory://checkpoint",
            payload=native_model,
        )
        config = YOLOModelConfig(framework="pytorch")
        yolo_cls.reset_mock()

        model = YOLOModelProvider().restore_checkpoint(config, checkpoint)

        yolo_cls.assert_not_called()
        assert model.native_model is native_model

    def test_unsupported_payload_raises(self) -> None:
        from bovi_yolo.models import YOLOModelConfig, YOLOModelProvider

        artifact = ResolvedModelArtifact[object](
            format="state-dict",
            source_uri="memory://artifact",
            payload={"weights": "unsupported"},
        )

        with pytest.raises(TypeError, match="callable native model"):
            YOLOModelProvider().load_artifact(
                YOLOModelConfig(framework="pytorch"),
                artifact,
            )

    def test_missing_resolved_file_raises_without_implicit_download(self) -> None:
        from bovi_yolo.models import YOLOModelConfig, YOLOModelProvider

        artifact = ResolvedModelArtifact[object](
            format="ultralytics-pt",
            source_uri="file:///tmp/missing-yolo-weights.pt",
            local_path=Path("/tmp/missing-yolo-weights.pt"),
        )

        with pytest.raises(ValueError, match="does not exist"):
            YOLOModelProvider().load_artifact(
                YOLOModelConfig(framework="pytorch"),
                artifact,
            )

    def test_unsupported_local_file_format_raises(self, tmp_path: Path) -> None:
        from bovi_yolo.models import YOLOModelConfig, YOLOModelProvider

        artifact_path = tmp_path / "model.onnx"
        artifact_path.touch()
        artifact = ResolvedModelArtifact[object](
            format="onnx",
            source_uri=artifact_path.as_uri(),
            local_path=artifact_path,
        )

        with pytest.raises(ValueError, match="Unsupported YOLO file format"):
            YOLOModelProvider().load_artifact(
                YOLOModelConfig(framework="pytorch"),
                artifact,
            )


def test_provider_registered() -> None:
    from bovi_yolo.models import YOLOModelProvider

    assert ModelProviderRegistry.get("yolo") is YOLOModelProvider
