from typing import ClassVar

import pytest
from bovi_core.config import ConfigNode
from bovi_core.ml.dataloaders.config import DataLoaderConfig, LoaderSettings
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class SourceSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str


class DatasetSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target: str


class ConcreteDataLoaderConfig(DataLoaderConfig):
    model_key: ClassVar[str] = "example"

    dataset: DatasetSettings
    source: SourceSettings
    dataloader: LoaderSettings = Field(default_factory=LoaderSettings)


class SplitOnlyDataLoaderConfig(DataLoaderConfig):
    model_key: ClassVar[str] = "split_only"


def test_dataloader_config_can_be_constructed_without_yaml():
    config = ConcreteDataLoaderConfig(
        split="train",
        dataset=DatasetSettings(target="yield"),
        source=SourceSettings(path="train.json"),
        dataloader=LoaderSettings(batch_size=16, shuffle=True, seed=7),
    )

    assert config.split == "train"
    assert config.dataloader.batch_size == 16


def test_from_config_combines_model_dataset_and_selected_split(config_setup):
    config_setup.experiment.models.example = ConfigNode(
        {
            "framework": "example",
            "dataset": {"target": "yield"},
            "dataloaders": {
                "train": {
                    "source": {"path": "train.json"},
                    "dataloader": {"batch_size": 8, "shuffle": True, "seed": 11},
                },
                "validation": {
                    "source": {"path": "validation.json"},
                },
            },
        }
    )

    result = ConcreteDataLoaderConfig.from_config(config_setup, split="train")

    assert result == ConcreteDataLoaderConfig(
        split="train",
        dataset=DatasetSettings(target="yield"),
        source=SourceSettings(path="train.json"),
        dataloader=LoaderSettings(batch_size=8, shuffle=True, seed=11),
    )


def test_from_config_rejects_split_level_framework(config_setup):
    config_setup.experiment.models.example = ConfigNode(
        {
            "framework": "example",
            "dataset": {"target": "yield"},
            "dataloaders": {
                "train": {
                    "framework": "sklearn",
                    "source": {"path": "train.json"},
                }
            },
        }
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ConcreteDataLoaderConfig.from_config(config_setup, split="train")


def test_from_config_does_not_require_model_wide_dataset(config_setup):
    config_setup.experiment.models.split_only = ConfigNode({"dataloaders": {"predict": {}}})

    result = SplitOnlyDataLoaderConfig.from_config(config_setup, split="predict")

    assert result == SplitOnlyDataLoaderConfig(split="predict")


@pytest.mark.parametrize("split", ["", "   "])
def test_split_must_not_be_empty(split):
    with pytest.raises(ValidationError):
        DataLoaderConfig(split=split)


def test_config_is_frozen_and_rejects_unknown_fields():
    config = DataLoaderConfig(split="train")

    with pytest.raises(ValidationError):
        config.split = "validation"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DataLoaderConfig.model_validate({"split": "train", "typo": True})


def test_from_config_requires_model_key(config_setup):
    with pytest.raises(TypeError, match="must define a model_key"):
        DataLoaderConfig.from_config(config_setup, split="train")


@pytest.mark.parametrize("batch_size", [0, -1])
def test_loader_settings_requires_positive_batch_size(batch_size):
    with pytest.raises(ValidationError):
        LoaderSettings(batch_size=batch_size)


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_loader_settings_rejects_seed_outside_uint32(seed):
    with pytest.raises(ValidationError):
        LoaderSettings(seed=seed)


def test_loader_settings_accepts_optional_seed_and_serializes():
    settings = LoaderSettings(batch_size=4, shuffle=False, seed=None)

    assert settings.model_dump() == {"batch_size": 4, "shuffle": False, "seed": None}
