"""Reject misspelled selected data settings at the example factory boundary."""

import pytest
from pydantic import ValidationError
from scikit_sgd import ScikitSGDDataLoaderConfig
from scikit_sgd.dataloaders.config import (
    JSONRecordsSourceConfig,
    ScikitLoaderSettings,
    TabularDatasetConfig,
)


@pytest.mark.parametrize(
    ("section", "location"),
    [
        ("dataset", ("dataset", "typo")),
        ("source", ("source", "typo")),
        ("dataloader", ("dataloader", "typo")),
        ("split", ("typo",)),
    ],
)
def test_config_rejects_unknown_data_keys(experiment_config, monkeypatch, section, location):
    config = experiment_config
    node = config.experiment.models.scikit_sgd
    split = node.dataloaders.train
    target = (
        node.dataset
        if section == "dataset"
        else (split if section == "split" else getattr(split, section))
    )
    monkeypatch.setattr(target, "typo", 1, raising=False)

    with pytest.raises(ValidationError) as error:
        ScikitSGDDataLoaderConfig.from_config(config, "train")

    assert (location, "extra_forbidden") in [
        (issue["loc"], issue["type"]) for issue in error.value.errors()
    ]


def test_dataloader_config_supports_direct_construction(tmp_path):
    config = ScikitSGDDataLoaderConfig(
        split="calibration",
        dataset=TabularDatasetConfig(target_name="yield"),
        source=JSONRecordsSourceConfig(type="json_records", path=tmp_path / "records.json"),
        dataloader=ScikitLoaderSettings(batch_size=8, shuffle=False, seed=7),
    )

    assert config.split == "calibration"
    assert config.dataloader.batch_size == 8
