"""Reject misspelled selected data settings at the example factory boundary."""

import pytest
from pydantic import ValidationError
from scikit_sgd import create_dataloader


@pytest.mark.parametrize(
    ("section", "location"),
    [
        ("dataset", ("dataset", "typo")),
        ("source", ("split", "source", "typo")),
        ("dataloader", ("split", "dataloader", "typo")),
        ("split", ("split", "typo")),
    ],
)
def test_factory_rejects_unknown_data_keys(
    experiment_config, model_config, monkeypatch, section, location
):
    config, definition = experiment_config, model_config
    node = config.experiment.models.scikit_sgd
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
