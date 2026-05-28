"""Tests for the herd profiles CRUD API."""

import asyncio

from bovi_api.database import get_session
from bovi_api.models import UploadedDataset
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


def test_csv_preview_rejects_non_csv_extension(client):
    response = client.post(
        "/herd-profiles/csv-preview",
        data={"organization_id": "1"},
        files={"file": ("data.xlsx", b"PK\x03\x04", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_csv_preview_rejects_unrecognised_columns(client):
    response = client.post(
        "/herd-profiles/csv-preview",
        data={"organization_id": "1"},
        files={"file": ("bad.csv", b"breed,farm\nHolstein,Farm1\n", "text/csv")},
    )
    assert response.status_code == 400
