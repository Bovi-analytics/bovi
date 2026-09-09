"""Reject misspelled selected data settings at the example factory boundary."""

import pytest
from pydantic import ValidationError
from pytorch_linear import create_dataloader

pytestmark = pytest.mark.torch


@pytest.mark.parametrize(
    ("section", "location"),
    [
        ("dataset", ("dataset", "typo")),
        ("source", ("split", "source", "typo")),
        ("dataloader", ("split", "dataloader", "typo")),
        ("split", ("split", "typo")),
    ],
)
def test_factory_rejects_unknown_data_keys(pipeline, monkeypatch, section, location):
    config, definition, _ = pipeline
    node = config.experiment.models.pytorch_linear
    split = node.dataloaders.train
    target = (
        node.dataset
        if section == "dataset"
        else (split if section == "split" else getattr(split, section))
    )
    monkeypatch.setattr(target, "typo", 1, raising=False)

    with pytest.raises(ValidationError) as error:
        create_dataloader(config, definition, "train")

    assert (location, "extra_forbidden") in [
        (issue["loc"], issue["type"]) for issue in error.value.errors()
    ]
