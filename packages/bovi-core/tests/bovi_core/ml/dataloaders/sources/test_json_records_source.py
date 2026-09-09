"""Shared JSON records, provenance, and transformed tabular pipeline tests."""

import json

import numpy as np
import pytest
from bovi_core.ml.dataloaders import JSONRecordsSource, SklearnDataLoader, TabularDataset
from bovi_core.ml.dataloaders.sources import TransformedSource
from bovi_core.ml.dataloaders.transforms import TransformRegistry


def test_json_records_source_preserves_order_metadata_and_independent_records(tmp_path):
    path = tmp_path / "records.json"
    records = [{"sample_id": "a", "nested": {"x": [1]}}, {"sample_id": "b"}]
    path.write_text(json.dumps(records), encoding="utf-8")
    source = JSONRecordsSource(path)

    assert len(source) == 2
    assert source.get_keys() == [0, 1]
    assert source.load_item("1") == records[1]
    assert source.get_metadata(-1) == {"index": 1, "path": str(path), "sample_id": "b"}
    source.load_item(0)["nested"]["x"].append(2)
    assert source.load_item(0) == records[0]


@pytest.mark.parametrize("records", [{}, [1], [None], "records"])
def test_json_records_source_rejects_non_record_arrays(tmp_path, records):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array of records"):
        JSONRecordsSource(path)


def test_json_records_source_empty_missing_and_invalid_json(tmp_path):
    path = tmp_path / "records.json"
    with pytest.raises(FileNotFoundError):
        JSONRecordsSource(path)
    path.write_text("[", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        JSONRecordsSource(path)
    path.write_text("[]", encoding="utf-8")
    assert JSONRecordsSource(path).get_keys() == []


def test_json_transforms_tabular_dataset_and_loader_share_examples(
    tmp_path, mock_dataloader_config
):
    path = tmp_path / "records.json"
    path.write_text(json.dumps([{"x": 8, "y": 3}, {"x": 16, "y": 5}]), encoding="utf-8")
    raw = JSONRecordsSource(path)
    transforms = TransformRegistry.from_config(
        [
            {"name": "numeric_scale", "params": {"factors": {"x": 2}}},
            {"name": "numeric_scale", "params": {"factors": {"x": 4}}},
        ]
    )
    dataset = TabularDataset(TransformedSource(raw, transforms), ("x",), "y")
    loader = SklearnDataLoader(
        dataset, mock_dataloader_config, model_name="test_model", batch_size=2, shuffle=False
    )
    batch = next(iter(loader))
    example = dataset.get_input_example(n_samples=2)
    assert isinstance(example, dict)
    np.testing.assert_array_equal(batch["features"]["x"], [1, 2])
    np.testing.assert_array_equal(example["features"]["x"], batch["features"]["x"])
    np.testing.assert_array_equal(example["labels"], batch["labels"])
    assert example["metadata"] == batch["metadata"]
    assert batch["metadata"][0]["path"] == str(path)
    assert "sample_id" not in batch["metadata"][0]
