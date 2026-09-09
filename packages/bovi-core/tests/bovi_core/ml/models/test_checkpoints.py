"""Local bundle publication and validation do not require a native ML framework."""

import json
import os
from pathlib import Path

import bovi_core.ml.models.checkpoints as checkpoints
import pytest
from bovi_core.ml.models.checkpoints import LocalCheckpointResolver, LocalCheckpointStore
from bovi_core.ml.models.resources import CheckpointReference


def write_bundle(directory: Path) -> None:
    (directory / "model.bin").write_bytes(b"weights")
    (directory / "features.json").write_text('["x"]')


def _resolve_path(reference: CheckpointReference) -> Path:
    resolved = LocalCheckpointResolver().resolve(reference)
    assert resolved.local_path is not None
    return resolved.local_path


def test_versions_are_immutable_and_roundtrip(tmp_path):
    store = LocalCheckpointStore(tmp_path)
    first = store.save("last", "native-test", write_bundle, entrypoint="model.bin")
    second = store.save("last", "native-test", write_bundle, entrypoint="model.bin")
    assert first.uri != second.uri
    resolved = LocalCheckpointResolver().resolve(first)
    assert resolved.local_path is not None
    assert resolved.local_path.read_bytes() == b"weights"
    assert resolved.metadata["resume_scope"] == "weights_only"
    manifest = json.loads((resolved.local_path.parent / "manifest.json").read_text())
    assert set(manifest["files"]) == {"model.bin", "features.json"}


def test_incomplete_write_keeps_previous_checkpoint(tmp_path):
    store = LocalCheckpointStore(tmp_path)
    first = store.save("last", "native-test", write_bundle, entrypoint="model.bin")

    def fail(directory):
        (directory / "model.bin").write_bytes(b"partial")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        store.save("last", "native-test", fail, entrypoint="model.bin")
    assert len(list(tmp_path.iterdir())) == 1
    assert _resolve_path(first).read_bytes() == b"weights"


@pytest.mark.parametrize("damage", ["missing", "corrupt", "extra", "manifest"])
def test_rejects_damaged_bundle(tmp_path, damage):
    reference = LocalCheckpointStore(tmp_path).save(
        "last", "native-test", write_bundle, entrypoint="model.bin"
    )
    root = _resolve_path(reference).parent
    if damage == "missing":
        (root / "features.json").unlink()
    elif damage == "corrupt":
        (root / "model.bin").write_bytes(b"broken")
    elif damage == "extra":
        (root / "unlisted").touch()
    else:
        (root / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum|files"):
        LocalCheckpointResolver().resolve(reference)


def test_rejects_wrong_format_and_incomplete_writer(tmp_path):
    store = LocalCheckpointStore(tmp_path)
    reference = store.save("last", "native-test", write_bundle, entrypoint="model.bin")
    with pytest.raises(ValueError, match="format"):
        LocalCheckpointResolver().resolve(reference.model_copy(update={"format": "wrong"}))
    with pytest.raises(ValueError, match="entrypoint"):
        store.save("last", "native-test", lambda _: None, entrypoint="model.bin")
    assert len(list(tmp_path.iterdir())) == 1


def test_manifest_cannot_escape_bundle(tmp_path):
    reference = LocalCheckpointStore(tmp_path).save(
        "last", "native-test", write_bundle, entrypoint="model.bin"
    )
    root = _resolve_path(reference).parent
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["entrypoint"] = "../outside"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="entrypoint"):
        LocalCheckpointResolver().resolve(reference.model_copy(update={"checksum": None}))


@pytest.mark.parametrize("after_publish", [False, True])
def test_directory_sync_failure_preserves_prior_reference(tmp_path, monkeypatch, after_publish):
    store = LocalCheckpointStore(tmp_path)
    first = store.save("last", "native-test", write_bundle, entrypoint="model.bin")
    original = checkpoints._sync_directory

    def fail(path):
        if (path == tmp_path) == after_publish:
            raise OSError("directory sync failed")
        original(path)

    monkeypatch.setattr(checkpoints, "_sync_directory", fail)
    with pytest.raises(OSError, match="directory sync failed"):
        store.save("last", "native-test", write_bundle, entrypoint="model.bin")
    assert _resolve_path(first).read_bytes() == b"weights"
    assert not list(tmp_path.glob(".pending-*"))
    assert len(list(tmp_path.iterdir())) == (2 if after_publish else 1)


def test_nested_directory_entries_are_synced_before_publication(tmp_path, monkeypatch):
    synced = []
    monkeypatch.setattr(checkpoints, "_sync_directory", synced.append)

    def write(root):
        (root / "nested").mkdir()
        (root / "nested" / "model.bin").write_bytes(b"weights")

    reference = LocalCheckpointStore(tmp_path).save(
        "last", "native-test", write, entrypoint="nested/model.bin"
    )
    assert synced[0].name == "nested"
    assert synced[1].name.startswith(".pending-")
    assert synced[2] == tmp_path
    assert _resolve_path(reference).read_bytes() == b"weights"


@pytest.mark.parametrize("precreate", [False, True])
def test_storage_ancestry_is_synced_before_return(tmp_path, monkeypatch, precreate):
    storage = tmp_path / "attempt" / "checkpoints"
    if precreate:
        storage.mkdir(parents=True)
    synced = []
    monkeypatch.setattr(checkpoints, "_sync_directory", synced.append)

    reference = LocalCheckpointStore(storage).save(
        "last", "native-test", write_bundle, entrypoint="model.bin"
    )

    assert synced[0].name.startswith(".pending-")
    assert synced[1:] == [storage, *storage.parents]
    assert _resolve_path(reference).read_bytes() == b"weights"


def test_ancestor_sync_failure_preserves_old_reference_and_retry_syncs_again(tmp_path, monkeypatch):
    storage = tmp_path / "attempt" / "checkpoints"
    store = LocalCheckpointStore(storage)
    first = store.save("last", "native-test", write_bundle, entrypoint="model.bin")
    original = checkpoints._sync_directory
    synced = []

    def fail(path):
        if path == storage.parent:
            raise OSError("ancestor sync failed")
        original(path)

    monkeypatch.setattr(checkpoints, "_sync_directory", fail)
    with pytest.raises(OSError, match="ancestor sync failed"):
        store.save("last", "native-test", write_bundle, entrypoint="model.bin")
    assert len(list(storage.iterdir())) == 2
    assert not list(storage.glob(".pending-*"))
    assert _resolve_path(first).read_bytes() == b"weights"

    def record(path):
        synced.append(path)
        original(path)

    monkeypatch.setattr(checkpoints, "_sync_directory", record)
    retried = store.save("last", "native-test", write_bundle, entrypoint="model.bin")
    assert synced[1:] == [storage, *storage.parents]
    assert _resolve_path(retried).read_bytes() == b"weights"


@pytest.mark.parametrize("entrypoint", ["", "../escape", "/absolute", "model\\name.bin"])
def test_invalid_entrypoint_is_rejected_before_writer(tmp_path, entrypoint):
    called = False

    def write(root):
        nonlocal called
        called = True
        write_bundle(root)

    with pytest.raises(ValueError, match="Invalid checkpoint file path"):
        LocalCheckpointStore(tmp_path).save("last", "native-test", write, entrypoint=entrypoint)
    assert not called
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(
    os.name != "posix", reason="Backslashes are literal filename characters on POSIX"
)
@pytest.mark.parametrize("filename", ["sidecar\\name.json", "nested\\dir/features.json"])
def test_unsupported_payload_path_cannot_publish_bundle(tmp_path, filename):
    store = LocalCheckpointStore(tmp_path)
    first = store.save("last", "native-test", write_bundle, entrypoint="model.bin")

    def write(root):
        write_bundle(root)
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]")

    with pytest.raises(ValueError, match="Invalid checkpoint file path"):
        store.save("last", "native-test", write, entrypoint="model.bin")
    assert len(list(tmp_path.iterdir())) == 1
    assert _resolve_path(first).read_bytes() == b"weights"
