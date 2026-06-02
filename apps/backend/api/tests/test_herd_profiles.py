"""Tests for the herd profiles CRUD API."""

import asyncio
from datetime import datetime

from bovi_api.database import get_session
from bovi_api.models import HerdProfile, StorageArtifact, UploadedDataset
from bovi_api.settings import Settings, get_settings
from sqlmodel import select

VALID_PROFILE = {
    "organization_id": 1,
    "name": "Test Herd",
    "description": "A test herd profile",
    "achieved_21_milk": 0.53,
    "achieved_305_milk": 0.50,
    "achieved_75_milk": 0.55,
    "achieved_milk": 0.41,
    "days_dry": 0.39,
    "days_in_milk": 0.44,
    "days_open": 0.38,
    "days_pregnant": 0.62,
    "historic_calving_interval": 0.54,
    "quality_sequence": 0.33,
}

PROFILE_FIELD_TO_STAT = {
    "achieved_21_milk": "Achieved21Milk",
    "achieved_305_milk": "Achieved305Milk",
    "achieved_75_milk": "Achieved75Milk",
    "achieved_milk": "AchievedMilk",
    "days_dry": "DaysDry",
    "days_in_milk": "DaysInMilk",
    "days_open": "DaysOpen",
    "days_pregnant": "DaysPregnant",
    "historic_calving_interval": "HistoricCalvingInterval",
    "quality_sequence": "QualitySequence",
}


def _profile_payload_from_upload(upload: dict, *, name: str) -> dict:
    return {
        "organization_id": 1,
        "name": name,
        "description": "Derived from uploaded dataset",
        **{field: upload["stats"][stat] for field, stat in PROFILE_FIELD_TO_STAT.items()},
    }


def test_create_profile(client):
    response = client.post("/herd-profiles/", json=VALID_PROFILE)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Herd"
    assert data["id"] is not None
    assert "created_at" in data  # may be None under SQLite


def test_list_profiles_empty(client):
    response = client.get("/herd-profiles/?organization_id=1")
    assert response.status_code == 200
    assert response.json() == []


def test_list_profiles_after_create(client):
    client.post("/herd-profiles/", json=VALID_PROFILE)
    response = client.get("/herd-profiles/?organization_id=1")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_get_profile(client):
    created = client.post("/herd-profiles/", json=VALID_PROFILE).json()
    response = client.get(f"/herd-profiles/{created['id']}")
    assert response.status_code == 200
    assert response.json()["name"] == "Test Herd"


def test_get_profile_not_found(client):
    response = client.get("/herd-profiles/999")
    assert response.status_code == 404


def test_update_profile(client):
    created = client.post("/herd-profiles/", json=VALID_PROFILE).json()
    updated = {**VALID_PROFILE, "name": "Updated Herd", "achieved_21_milk": 0.70}
    response = client.put(f"/herd-profiles/{created['id']}", json=updated)
    assert response.status_code == 200
    assert response.json()["name"] == "Updated Herd"
    assert response.json()["achieved_21_milk"] == 0.70


def test_delete_profile(client):
    created = client.post("/herd-profiles/", json=VALID_PROFILE).json()
    response = client.delete(f"/herd-profiles/{created['id']}")
    assert response.status_code == 204
    assert client.get(f"/herd-profiles/{created['id']}").status_code == 404


def test_duplicate_name_returns_409(client):
    client.post("/herd-profiles/", json=VALID_PROFILE)
    response = client.post("/herd-profiles/", json=VALID_PROFILE)
    assert response.status_code == 409


def test_invalid_stat_value_returns_422(client):
    invalid = {**VALID_PROFILE, "achieved_21_milk": 1.5}  # exceeds 1.0
    response = client.post("/herd-profiles/", json=invalid)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# CSV preview endpoint
# ---------------------------------------------------------------------------

SAMPLE_CSV = (
    b"Achieved21Milk,Achieved305Milk,Achieved75Milk,AchievedMilk,"
    b"DaysDry,DaysInMilk,DaysOpen,DaysPregnant,HistoricCalvingInterval,QualitySequence\n"
    b"25.0,9000.0,28.0,10000.0,60.0,180.0,100.0,150.0,420.0,0.8\n"
)


def test_csv_preview_returns_normalized_stats(client):
    response = client.post(
        "/herd-profiles/csv-preview",
        data={"organization_id": "1"},
        files={"file": ("herd.csv", SAMPLE_CSV, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "stats" in data
    assert data["format_detected"] == "aggregated"
    assert data["upload_id"]
    assert data["row_count"] == 1
    for value in data["stats"].values():
        assert 0.0 <= value <= 1.0

    blob_paths = set(client.app.state.blob_container_client.store)
    assert any(path.endswith("/raw/upload.csv") for path in blob_paths)
    assert any(path.endswith("/parsed/stats.json.gz") for path in blob_paths)

    override = client.app.dependency_overrides[get_session]

    async def _uploaded_dataset() -> UploadedDataset:
        async for session in override():
            result = await session.execute(select(UploadedDataset))
            return result.scalars().one()
        raise AssertionError("session override did not yield")

    dataset = asyncio.run(_uploaded_dataset())
    assert dataset.id == data["upload_id"]
    assert dataset.user_id == 1
    assert dataset.organization_id == 1
    assert dataset.row_count == 1
    assert dataset.original_filename == "herd.csv"

    uploads = client.get("/uploaded-datasets?organization_id=1")
    assert uploads.status_code == 200
    upload = uploads.json()[0]
    assert upload["id"] == data["upload_id"]
    assert upload["user_name"] == "Test User"
    assert upload["organization_name"] == "Test Organization"

    detail = client.get(f"/uploaded-datasets/{data['upload_id']}")
    assert detail.status_code == 200
    assert detail.json()["stats"] == data["stats"]


def test_csv_preview_reuses_identical_dataset_artifacts_and_archives_uploads(client):
    first = client.post(
        "/herd-profiles/csv-preview",
        data={"organization_id": "1"},
        files={"file": ("herd.csv", SAMPLE_CSV, "text/csv")},
    )
    second = client.post(
        "/herd-profiles/csv-preview",
        data={"organization_id": "1"},
        files={"file": ("renamed.csv", SAMPLE_CSV, "text/csv")},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["upload_id"] != second.json()["upload_id"]
    assert len(client.app.state.blob_container_client.store) == 3

    override = client.app.dependency_overrides[get_session]

    async def _storage_state() -> tuple[list[UploadedDataset], list[StorageArtifact]]:
        async for session in override():
            datasets = (await session.execute(select(UploadedDataset))).scalars().all()
            artifacts = (await session.execute(select(StorageArtifact))).scalars().all()
            return datasets, artifacts
        raise AssertionError("session override did not yield")

    datasets, artifacts = asyncio.run(_storage_state())
    assert len(datasets) == 2
    assert len(artifacts) == 3
    assert datasets[0].raw_file_artifact_id == datasets[1].raw_file_artifact_id
    assert datasets[0].stats_artifact_id == datasets[1].stats_artifact_id

    assert client.delete(f"/uploaded-datasets/{first.json()['upload_id']}").status_code == 204
    datasets, artifacts = asyncio.run(_storage_state())
    assert len(datasets) == 2
    archived = next(dataset for dataset in datasets if dataset.id == first.json()["upload_id"])
    assert isinstance(archived.deleted_at, datetime)
    assert archived.deleted_by_user_id == 1
    assert len(artifacts) == 3
    assert len(client.app.state.blob_container_client.store) == 3
    uploads = client.get("/uploaded-datasets?organization_id=1")
    assert uploads.status_code == 200
    assert [upload["id"] for upload in uploads.json()] == [second.json()["upload_id"]]
    assert client.get(f"/uploaded-datasets/{first.json()['upload_id']}").status_code == 404

    assert client.delete(f"/uploaded-datasets/{second.json()['upload_id']}").status_code == 204
    datasets, artifacts = asyncio.run(_storage_state())
    assert len(datasets) == 2
    assert all(dataset.deleted_at is not None for dataset in datasets)
    assert len(artifacts) == 3
    assert len(client.app.state.blob_container_client.store) == 3
    uploads = client.get("/uploaded-datasets?organization_id=1")
    assert uploads.status_code == 200
    assert uploads.json() == []


def test_uploaded_dataset_delete_impact_lists_linked_and_matching_profiles(client):
    upload = client.post(
        "/herd-profiles/csv-preview",
        data={"organization_id": "1"},
        files={"file": ("herd.csv", SAMPLE_CSV, "text/csv")},
    )
    assert upload.status_code == 200
    upload_data = upload.json()

    linked_payload = {
        **_profile_payload_from_upload(upload_data, name="Linked profile"),
        "source_uploaded_dataset_id": upload_data["upload_id"],
    }
    linked = client.post("/herd-profiles/", json=linked_payload)
    assert linked.status_code == 201
    assert linked.json()["source_uploaded_dataset_id"] == upload_data["upload_id"]
    unrelated = client.post(
        "/herd-profiles/",
        json={**VALID_PROFILE, "name": "Unrelated profile"},
    )
    assert unrelated.status_code == 201

    override = client.app.dependency_overrides[get_session]

    async def _seed_legacy_matching_profile() -> None:
        async for session in override():
            session.add(
                HerdProfile(
                    id=902,
                    user_id=1,
                    **_profile_payload_from_upload(upload_data, name="Legacy matching profile"),
                )
            )
            await session.commit()
            break

    asyncio.run(_seed_legacy_matching_profile())

    impact = client.get(f"/uploaded-datasets/{upload_data['upload_id']}/delete-impact")

    assert impact.status_code == 200
    profiles = impact.json()["herd_profiles"]
    assert {profile["name"] for profile in profiles} == {
        "Linked profile",
        "Legacy matching profile",
    }
    assert {profile["reference_type"] for profile in profiles} == {"linked", "matching_stats"}

    delete_response = client.delete(f"/uploaded-datasets/{upload_data['upload_id']}")
    assert delete_response.status_code == 204
    remaining_profiles = client.get("/herd-profiles/?organization_id=1")
    assert remaining_profiles.status_code == 200
    assert [profile["name"] for profile in remaining_profiles.json()] == ["Unrelated profile"]


def test_csv_preview_rejects_non_csv_extension(client):
    response = client.post(
        "/herd-profiles/csv-preview",
        data={"organization_id": "1"},
        files={"file": ("data.xlsx", b"PK\x03\x04", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_csv_preview_rejects_large_file_with_clear_message(client):
    client.app.dependency_overrides[get_settings] = lambda: Settings(upload_max_bytes=16)

    response = client.post(
        "/herd-profiles/csv-preview",
        data={"organization_id": "1"},
        files={"file": ("oversized.csv", SAMPLE_CSV, "text/csv")},
    )

    assert response.status_code == 413
    detail = response.json()["detail"]
    assert "oversized.csv" in detail
    assert "16 bytes upload limit" in detail
    assert "Split the file into smaller CSV files" in detail


def test_csv_preview_rejects_unrecognised_columns(client):
    response = client.post(
        "/herd-profiles/csv-preview",
        data={"organization_id": "1"},
        files={"file": ("bad.csv", b"breed,farm\nHolstein,Farm1\n", "text/csv")},
    )
    assert response.status_code == 400
