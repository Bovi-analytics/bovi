"""Tests for preset dataset generation blob paths."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_preset_datasets.py"
_SPEC = importlib.util.spec_from_file_location("generate_preset_datasets", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
generator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(generator)


def test_generator_reads_raw_data_from_canonical_blob_prefix():
    assert generator.DATASET_CONFIGS["aurora"]["blob_path"] == "data/raw/AuroraTDM23_26.csv"
    assert (
        generator.DATASET_CONFIGS["sunnyside"]["blob_path"]
        == "data/raw/MilkRecordingsSunnyside.csv"
    )


def test_generator_writes_presets_to_canonical_blob_prefix(monkeypatch):
    uploaded: list[str] = []

    class FakeBlobClient:
        def __init__(self, path: str):
            self.path = path

        def upload_blob(self, data, overwrite=False, content_type=None):
            uploaded.append(self.path)

    class FakeServiceClient:
        def get_blob_client(self, container, blob_path):
            return FakeBlobClient(blob_path)

    class FakeDataFrame:
        def copy(self):
            return self

    monkeypatch.setattr(
        generator,
        "_read_blob_csv",
        lambda client, container, blob_path: FakeDataFrame(),
    )
    monkeypatch.setattr(generator, "_build_lactations", lambda df, config, period: [])
    monkeypatch.setattr(
        generator,
        "_build_icar_preset",
        lambda client, container: b'{"cow_count": 0}',
    )
    monkeypatch.setattr(
        generator.BlobServiceClient,
        "from_connection_string",
        lambda conn_str: FakeServiceClient(),
    )
    monkeypatch.setenv("CONNECTION_STRING", "UseDevelopmentStorage=true")
    monkeypatch.setenv("STORAGE_ACCOUNT_CONTAINER_ICAR", "testcontainer")

    generator.main(dry_run=False)

    assert "data/datasets/presets/aurora/small_recent.json" in uploaded
    assert "data/datasets/presets/sunnyside/large_mixed.json" in uploaded
    assert "data/datasets/presets/icar/full.json" in uploaded
    assert not any(path.startswith("preset-datasets/") for path in uploaded)
