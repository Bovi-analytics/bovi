"""Admin overview route tests."""

import asyncio
import json
from datetime import datetime, timedelta, timezone

from bovi_api.database import get_session
from bovi_api.models import Challenge, HerdProfile, Submission, UploadedDataset


def _profile_kwargs(name: str, user_id: int, organization_id: int, created_at: datetime) -> dict:
    return {
        "name": name,
        "description": "",
        "achieved_21_milk": 0.5,
        "achieved_305_milk": 0.5,
        "achieved_75_milk": 0.5,
        "achieved_milk": 0.5,
        "days_dry": 0.5,
        "days_in_milk": 0.5,
        "days_open": 0.5,
        "days_pregnant": 0.5,
        "historic_calving_interval": 0.5,
        "quality_sequence": 0.5,
        "user_id": user_id,
        "organization_id": organization_id,
        "created_at": created_at,
    }


def _seed_admin_items(client) -> None:
    override = client.app.dependency_overrides[get_session]
    base = datetime(2026, 5, 28, 9, 0, tzinfo=timezone.utc)

    async def _seed() -> None:
        async for session in override():
            session.add(
                Challenge(
                    id=101,
                    dataset="upload",
                    size="custom",
                    period="custom",
                    source="upload",
                    name="Alpha challenge",
                    cow_metadata={"cow1": {"parity": 1, "dim": [50], "milk_kg": [25.0]}},
                    actual_yields={"cow1": 8000.0},
                    row_count=1,
                    cow_count=1,
                    user_id=1,
                    organization_id=1,
                    created_at=base,
                )
            )
            session.add(
                Challenge(
                    id=202,
                    dataset="icar",
                    size="full",
                    period="all",
                    source="preset",
                    name="Beta challenge",
                    cow_metadata={"cow2": {"parity": 2, "dim": [60], "milk_kg": [30.0]}},
                    actual_yields={"cow2": 9000.0},
                    row_count=1,
                    cow_count=1,
                    user_id=2,
                    organization_id=2,
                    created_at=base + timedelta(hours=1),
                )
            )
            session.add(
                Submission(
                    id=301,
                    challenge_id=101,
                    submission_type="own_method",
                    model_type=None,
                    benchmark_model="tim",
                    organization="Alpha submitted method",
                    calculation_method="farm sheet",
                    submitted_yields={"cow1": 8050.0},
                    bovi_yields={"cow1": 7990.0},
                    stats={
                        "version": 2,
                        "challenger_vs_aly": {
                            "overall": {
                                "pearson": 0.9,
                                "rmse": 12.3,
                                "mae": 10.0,
                                "mape": 1.0,
                                "n": 1,
                            }
                        },
                        "failed_count": 1,
                    },
                    failed_cow_ids=["cow-bad"],
                    row_count=2,
                    submitted_yield_count=1,
                    benchmark_yield_count=1,
                    failed_count=1,
                    user_id=1,
                    organization_id=1,
                    created_at=base + timedelta(hours=2),
                )
            )
            session.add(
                UploadedDataset(
                    id="upload-1",
                    name="Beta herd upload",
                    dataset_type="herd_profile",
                    format_detected="icar_test_day",
                    user_id=2,
                    organization_id=2,
                    original_filename="beta.csv",
                    row_count=42,
                    cow_count=7,
                    columns=["TestId", "milk_kg"],
                    column_mapping={},
                    warnings=["Missing optional parity column."],
                    stats_summary={"days_in_milk": 0.5},
                    raw_stats_summary={"days_in_milk": 120},
                    uploaded_at=base + timedelta(hours=3),
                )
            )
            session.add(
                HerdProfile(
                    id=401,
                    **_profile_kwargs(
                        "Alpha herd profile",
                        user_id=1,
                        organization_id=1,
                        created_at=base + timedelta(hours=4),
                    ),
                )
            )
            await session.commit()
            break

    asyncio.run(_seed())


def test_admin_overview_requires_global_admin(client, multi_org_auth):
    _seed_admin_items(client)
    multi_org_auth.as_user("org1_owner")

    response = client.get("/admin/submissions-overview")

    assert response.status_code == 403


def test_admin_overview_returns_all_categories_without_raw_payloads(client, multi_org_auth):
    _seed_admin_items(client)
    multi_org_auth.as_user("admin")

    response = client.get("/admin/submissions-overview?organization_id=all")

    assert response.status_code == 200
    body = response.json()
    assert body["kpis"]["total_items"] == 5
    assert body["kpis"]["organizations"] == 2
    assert body["kpis"]["users"] == 2
    assert body["kpis"]["benchmark_submissions"] == 1
    assert body["kpis"]["benchmark_challenges"] == 2
    assert body["kpis"]["herd_dataset_uploads"] == 1
    assert body["kpis"]["herd_profiles"] == 1
    assert body["kpis"]["failed_items"] == 2
    assert {category["item_type"] for category in body["by_category"]} == {
        "benchmark_submission",
        "benchmark_challenge",
        "herd_dataset_upload",
        "herd_profile",
    }
    assert {org["organization_name"] for org in body["by_organization"]} == {
        "Test Organization",
        "Other Organization",
    }
    encoded = json.dumps(body)
    assert "cow_metadata" not in encoded
    assert "submitted_yields" not in encoded
    assert "bovi_yields" not in encoded
    assert "actual_yields" not in encoded


def test_admin_overview_filters_category_organization_user_and_search(client, multi_org_auth):
    _seed_admin_items(client)
    multi_org_auth.as_user("admin")

    by_org = client.get("/admin/submissions-overview?organization_id=2")
    assert by_org.status_code == 200
    assert {item["organization_name"] for item in by_org.json()["items"]} == {"Other Organization"}

    uploads = client.get("/admin/submissions-overview?category=herd_dataset_upload")
    assert uploads.status_code == 200
    assert [item["title"] for item in uploads.json()["items"]] == ["Beta herd upload"]

    by_user = client.get("/admin/submissions-overview?user_id=1")
    assert by_user.status_code == 200
    assert {item["user_email"] for item in by_user.json()["items"]} == {"user@example.test"}

    search = client.get("/admin/submissions-overview?q=farm%20sheet")
    assert search.status_code == 200
    assert [item["item_type"] for item in search.json()["items"]] == ["benchmark_submission"]
