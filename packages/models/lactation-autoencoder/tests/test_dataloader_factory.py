"""Tests for config-driven lactation dataloader construction."""

from bovi_core.config import Config
from bovi_core.ml.dataloaders import SklearnDataLoader
from lactation_autoencoder.dataloaders import create_dataloader


def test_create_dataloader_composes_inference_pipeline() -> None:
    Config.reset()
    config = Config(
        experiment_name="lactation_autoencoder",
        project_name="lactation-autoencoder",
    )

    loader = create_dataloader(config, split="inference", batch_size=1)
    batch = next(iter(loader))

    assert isinstance(loader, SklearnDataLoader)
    assert len(batch["features"]["milk"]) == 1
    assert batch["labels"].shape == (1, 304)
