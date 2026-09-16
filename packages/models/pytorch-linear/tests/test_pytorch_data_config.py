"""Reject misspelled selected data settings at the example factory boundary."""

import pytest
from pydantic import ValidationError
from pytorch_linear import PyTorchLinearDataLoaderConfig
from pytorch_linear.dataloaders.config import (
    JSONRecordsSourceConfig,
    PyTorchLoaderSettings,
    TabularDatasetConfig,
)

pytestmark = pytest.mark.torch


@pytest.mark.parametrize(
    ("section", "location"),
    [
        ("dataset", ("dataset", "typo")),
        ("source", ("source", "typo")),
        ("dataloader", ("dataloader", "typo")),
        ("split", ("typo",)),
    ],
)
def test_config_rejects_unknown_data_keys(pipeline, monkeypatch, section, location):
    config, _, _ = pipeline
    node = config.experiment.models.pytorch_linear
    split = node.dataloaders.train
    target = (
        node.dataset
        if section == "dataset"
        else (split if section == "split" else getattr(split, section))
    )
    monkeypatch.setattr(target, "typo", 1, raising=False)

    with pytest.raises(ValidationError) as error:
        PyTorchLinearDataLoaderConfig.from_config(config, "train")

    assert (location, "extra_forbidden") in [
        (issue["loc"], issue["type"]) for issue in error.value.errors()
    ]


def test_dataloader_config_supports_direct_construction(tmp_path):
    config = PyTorchLinearDataLoaderConfig(
        split="calibration",
        dataset=TabularDatasetConfig(target_name="y"),
        source=JSONRecordsSourceConfig(type="json_records", path=tmp_path / "records.json"),
        dataloader=PyTorchLoaderSettings(batch_size=8, shuffle=False, seed=None),
    )

    assert config.split == "calibration"
    assert config.dataloader.num_workers == 0
