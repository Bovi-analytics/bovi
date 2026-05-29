"""Authentication and organization authorization tests."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import cast

from bovi_api.auth import CurrentUser, require_auth
from bovi_api.database import get_session
from bovi_api.models import (
    Challenge,
    HerdProfile,
    Organization,
    OrganizationInvite,
    OrganizationMembership,
    Submission,
    UploadedDataset,
    User,
)
from bovi_api.routes.organizations import _token_hash, preview_invite
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select


def test_auth_me_returns_current_user(client):
    response = client.get("/auth/me")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 1
    assert body["entra_tenant_id"] == "test-tenant"
    assert body["account_type"] == "entra"
    assert body["email"] == "user@example.test"
    assert body["is_admin"] is False
    assert body["organizations"] == [{"id": 1, "name": "Test Organization", "role": "Owner"}]


def test_protected_route_propagates_auth_dependency_failure(client):
    async def reject_auth():
        raise HTTPException(status_code=401, detail="Authentication required.")

    client.app.dependency_overrides[require_auth] = reject_auth
    response = client.get("/benchmark/challenges")

    assert response.status_code == 401


def test_user_only_sees_challenges_for_their_organization(client):
    override = client.app.dependency_overrides[get_session]

    async def _seed() -> None:
        async for session in override():
            session.add(Organization(id=2, name="Other Organization"))
            session.add(
                User(
                    id=99,
                    entra_tenant_id="other-tenant",
                    entra_oid="other-user-oid",
                    account_type="entra",
                    email="other@example.test",
                    name="Other User",
                )
            )
            await session.commit()
            session.add(
                Challenge(
                    dataset="icar",
                    size="full",
                    period="all",
                    source="preset",
                    name="visible",
                    cow_metadata={"cow1": {"parity": 1, "dim": [50], "milk_kg": [25.0]}},
                    reference_yields=None,
                    actual_yields={"cow1": 8000.0},
                    user_id=1,
                    organization_id=1,
                )
            )
            session.add(
                Challenge(
                    dataset="icar",
                    size="full",
                    period="all",
                    source="preset",
                    name="hidden",
                    cow_metadata={"cow2": {"parity": 1, "dim": [50], "milk_kg": [25.0]}},
                    reference_yields=None,
                    actual_yields={"cow2": 8000.0},
                    user_id=99,
                    organization_id=2,
                )
            )
            await session.commit()
            break

    asyncio.run(_seed())

    response = client.get("/benchmark/challenges?organization_id=1")

    assert response.status_code == 200
    assert [challenge["name"] for challenge in response.json()] == ["visible"]


def test_admin_can_see_challenges_for_all_organizations(client):
    async def override_admin():
        return CurrentUser(
            id=7,
            entra_tenant_id="admin-tenant",
            entra_oid="admin-oid",
            account_type="entra",
            email="admin@example.test",
            name="Admin User",
            roles=["Admin"],
            is_admin=True,
            organizations=[],
        )

    client.app.dependency_overrides[require_auth] = override_admin
    override = client.app.dependency_overrides[get_session]

    async def _seed() -> None:
        async for session in override():
            session.add(Organization(id=2, name="Other Organization"))
            session.add(
                User(
                    id=2,
                    entra_tenant_id="other-tenant",
                    entra_oid="other-user-oid",
                    account_type="entra",
                    email="other@example.test",
                    name="Other User",
                )
            )
            await session.commit()
            for index, organization_id in enumerate([1, 2], start=1):
                session.add(
                    Challenge(
                        dataset="icar",
                        size="full",
                        period="all",
                        source="preset",
                        name=f"challenge-{index}",
                        cow_metadata={f"cow{index}": {"parity": 1, "dim": [50], "milk_kg": [25.0]}},
                        reference_yields=None,
                        actual_yields={f"cow{index}": 8000.0},
                        user_id=index,
                        organization_id=organization_id,
                    )
                )
            await session.commit()
            break

    asyncio.run(_seed())

    response = client.get("/benchmark/challenges?organization_id=all")

    assert response.status_code == 200
    assert {challenge["name"] for challenge in response.json()} == {"challenge-1", "challenge-2"}


def test_user_cannot_export_other_organization_challenge(client):
    override = client.app.dependency_overrides[get_session]

    async def _seed() -> None:
        async for session in override():
            session.add(Organization(id=2, name="Other Organization"))
            session.add(
                User(
                    id=99,
                    entra_tenant_id="other-tenant",
                    entra_oid="other-user-oid",
                    account_type="entra",
                    email="other@example.test",
                    name="Other User",
                )
            )
            await session.commit()
            session.add(
                Challenge(
                    id=44,
                    dataset="icar",
                    size="full",
                    period="all",
                    source="preset",
                    name="hidden",
                    cow_metadata={"cow2": {"parity": 1, "dim": [50], "milk_kg": [25.0]}},
                    reference_yields=None,
                    actual_yields={"cow2": 8000.0},
                    user_id=99,
                    organization_id=2,
                )
            )
            await session.commit()
            break

    asyncio.run(_seed())

    response = client.get("/benchmark/challenges/44/export")

    assert response.status_code == 404


def test_multi_org_fixture_enforces_record_isolation(client, multi_org_auth):
    override = client.app.dependency_overrides[get_session]

    async def _seed() -> None:
        async for session in override():
            session.add(
                Challenge(
                    id=10,
                    dataset="icar",
                    size="full",
                    period="all",
                    source="preset",
                    name="org1 challenge",
                    cow_metadata={"cow1": {"parity": 1, "dim": [50], "milk_kg": [25.0]}},
                    reference_yields=None,
                    actual_yields={"cow1": 8000.0},
                    user_id=1,
                    organization_id=1,
                )
            )
            session.add(
                Challenge(
                    id=20,
                    dataset="icar",
                    size="full",
                    period="all",
                    source="preset",
                    name="org2 challenge",
                    cow_metadata={"cow2": {"parity": 1, "dim": [60], "milk_kg": [26.0]}},
                    reference_yields=None,
                    actual_yields={"cow2": 8100.0},
                    user_id=2,
                    organization_id=2,
                )
            )
            session.add(
                Submission(
                    id=10,
                    challenge_id=10,
                    submission_type="bovi_model",
                    model_type="wood",
                    benchmark_model="tim",
                    organization="Org 1 submission",
                    submitted_yields={"cow1": 7900.0},
                    bovi_yields={"cow1": 8050.0},
                    stats={"version": 2},
                    failed_cow_ids=[],
                    user_id=1,
                    organization_id=1,
                )
            )
            session.add(
                Submission(
                    id=20,
                    challenge_id=20,
                    submission_type="bovi_model",
                    model_type="wood",
                    benchmark_model="tim",
                    organization="Org 2 submission",
                    submitted_yields={"cow2": 8000.0},
                    bovi_yields={"cow2": 8120.0},
                    stats={"version": 2},
                    failed_cow_ids=[],
                    user_id=2,
                    organization_id=2,
                )
            )
            session.add(
                HerdProfile(
                    id=10,
                    name="Org 1 profile",
                    description="",
                    achieved_21_milk=0.5,
                    achieved_305_milk=0.5,
                    achieved_75_milk=0.5,
                    achieved_milk=0.5,
                    days_dry=0.5,
                    days_in_milk=0.5,
                    days_open=0.5,
                    days_pregnant=0.5,
                    historic_calving_interval=0.5,
                    quality_sequence=0.5,
                    user_id=1,
                    organization_id=1,
                )
            )
            session.add(
                HerdProfile(
                    id=20,
                    name="Org 2 profile",
                    description="",
                    achieved_21_milk=0.5,
                    achieved_305_milk=0.5,
                    achieved_75_milk=0.5,
                    achieved_milk=0.5,
                    days_dry=0.5,
                    days_in_milk=0.5,
                    days_open=0.5,
                    days_pregnant=0.5,
                    historic_calving_interval=0.5,
                    quality_sequence=0.5,
                    user_id=2,
                    organization_id=2,
                )
            )
            await session.commit()
            break

    asyncio.run(_seed())

    multi_org_auth.as_user("org1_owner")
    assert [c["name"] for c in client.get("/benchmark/challenges?organization_id=1").json()] == [
        "org1 challenge"
    ]
    assert client.get("/benchmark/challenges?organization_id=2").status_code == 404
    assert client.get("/benchmark/challenges?organization_id=all").status_code == 403
    assert client.get("/benchmark/challenges/20").status_code == 404
    assert client.get("/benchmark/challenges/20/export").status_code == 404
    assert [
        s["organization"] for s in client.get("/benchmark/submissions?organization_id=1").json()
    ] == ["Org 1 submission"]
    assert client.get("/benchmark/submissions?organization_id=2").status_code == 404
    assert client.get("/benchmark/submissions/20").status_code == 404
    assert client.get("/benchmark/submissions/20/report").status_code == 404
    assert [p["name"] for p in client.get("/herd-profiles/?organization_id=1").json()] == [
        "Org 1 profile"
    ]
    assert client.get("/herd-profiles/?organization_id=2").status_code == 404
    assert client.get("/herd-profiles/20").status_code == 404
    assert (
        client.post(
            "/benchmark/challenges/saved-dataset",
            json={
                "name": "foreign create",
                "organization_id": 2,
                "cow_metadata": {"cow2": {"parity": 1, "dim": [60], "milk_kg": [26.0]}},
                "actual_yields": {"cow2": 8100.0},
            },
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/benchmark/challenges/upload",
            data={"name": "foreign upload", "organization_id": "2"},
            files={
                "test_day_csv": (
                    "test_day.csv",
                    b"TestId,parity,dim,milk_kg\ncow2,1,60,26.0\n",
                    "text/csv",
                ),
                "actual_yields_csv": (
                    "actual_yields.csv",
                    b"TestId,LactationYield\ncow2,8100.0\n",
                    "text/csv",
                ),
            },
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/herd-profiles/",
            json={
                "organization_id": 2,
                "name": "foreign profile",
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
            },
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/herd-profiles/csv-preview",
            data={"organization_id": "2"},
            files={
                "file": (
                    "herd.csv",
                    b"Achieved21Milk,Achieved305Milk,Achieved75Milk,AchievedMilk,"
                    b"DaysDry,DaysInMilk,DaysOpen,DaysPregnant,"
                    b"HistoricCalvingInterval,QualitySequence\n"
                    b"25,9000,28,10000,60,180,100,150,420,0.8\n",
                    "text/csv",
                )
            },
        ).status_code
        == 404
    )

    multi_org_auth.as_user("org2_owner")
    assert [c["name"] for c in client.get("/benchmark/challenges?organization_id=2").json()] == [
        "org2 challenge"
    ]
    assert client.get("/benchmark/challenges?organization_id=1").status_code == 404
    assert client.get("/herd-profiles/10").status_code == 404

    multi_org_auth.as_user("orgless")
    assert client.get("/benchmark/challenges?organization_id=1").status_code == 404
    assert client.get("/benchmark/challenges").status_code == 422

    multi_org_auth.as_user("admin")
    assert {c["name"] for c in client.get("/benchmark/challenges?organization_id=all").json()} == {
        "org1 challenge",
        "org2 challenge",
    }
    assert {
        s["organization"] for s in client.get("/benchmark/submissions?organization_id=all").json()
    } == {"Org 1 submission", "Org 2 submission"}
    assert {p["name"] for p in client.get("/herd-profiles/?organization_id=all").json()} == {
        "Org 1 profile",
        "Org 2 profile",
    }
    assert client.get("/benchmark/challenges/20").status_code == 200
    assert client.get("/benchmark/challenges/20/export").status_code == 200
    assert client.get("/benchmark/submissions/20").status_code == 200
    assert client.get("/herd-profiles/20").status_code == 200
    assert (
        client.post(
            "/benchmark/challenges/saved-dataset",
            json={
                "name": "admin org2 create",
                "organization_id": 2,
                "cow_metadata": {"cow3": {"parity": 1, "dim": [70], "milk_kg": [27.0]}},
                "actual_yields": {"cow3": 8200.0},
            },
        ).status_code
        == 201
    )
    assert (
        client.post(
            "/herd-profiles/csv-preview",
            data={"organization_id": "2"},
            files={
                "file": (
                    "admin-herd.csv",
                    b"Achieved21Milk,Achieved305Milk,Achieved75Milk,AchievedMilk,"
                    b"DaysDry,DaysInMilk,DaysOpen,DaysPregnant,"
                    b"HistoricCalvingInterval,QualitySequence\n"
                    b"25,9000,28,10000,60,180,100,150,420,0.8\n",
                    "text/csv",
                )
            },
        ).status_code
        == 200
    )


def test_org_members_can_filter_colleague_owned_records(client, multi_org_auth):
    override = client.app.dependency_overrides[get_session]

    async def _seed() -> None:
        async for session in override():
            session.add(
                User(
                    id=4,
                    entra_tenant_id="test-tenant",
                    entra_oid="colleague-oid",
                    account_type="entra",
                    email="colleague@example.test",
                    name="Colleague User",
                )
            )
            await session.commit()
            session.add(OrganizationMembership(user_id=4, organization_id=1, role="Member"))
            session.add(
                Challenge(
                    id=101,
                    dataset="icar",
                    size="full",
                    period="all",
                    source="preset",
                    name="mine",
                    cow_metadata={"cow1": {"parity": 1, "dim": [50], "milk_kg": [25.0]}},
                    reference_yields=None,
                    actual_yields={"cow1": 8000.0},
                    user_id=1,
                    organization_id=1,
                )
            )
            session.add(
                Challenge(
                    id=102,
                    dataset="icar",
                    size="full",
                    period="all",
                    source="preset",
                    name="colleague",
                    cow_metadata={"cow2": {"parity": 1, "dim": [60], "milk_kg": [26.0]}},
                    reference_yields=None,
                    actual_yields={"cow2": 8100.0},
                    user_id=4,
                    organization_id=1,
                )
            )
            session.add(
                HerdProfile(
                    id=101,
                    name="My profile",
                    description="",
                    achieved_21_milk=0.5,
                    achieved_305_milk=0.5,
                    achieved_75_milk=0.5,
                    achieved_milk=0.5,
                    days_dry=0.5,
                    days_in_milk=0.5,
                    days_open=0.5,
                    days_pregnant=0.5,
                    historic_calving_interval=0.5,
                    quality_sequence=0.5,
                    user_id=1,
                    organization_id=1,
                )
            )
            session.add(
                HerdProfile(
                    id=102,
                    name="Colleague profile",
                    description="",
                    achieved_21_milk=0.5,
                    achieved_305_milk=0.5,
                    achieved_75_milk=0.5,
                    achieved_milk=0.5,
                    days_dry=0.5,
                    days_in_milk=0.5,
                    days_open=0.5,
                    days_pregnant=0.5,
                    historic_calving_interval=0.5,
                    quality_sequence=0.5,
                    user_id=4,
                    organization_id=1,
                )
            )
            session.add(
                UploadedDataset(
                    id="dataset-mine",
                    name="Mine upload",
                    dataset_type="herd_profile",
                    format_detected="icar_test_day",
                    user_id=1,
                    organization_id=1,
                    original_filename="mine.csv",
                    row_count=1,
                    cow_count=1,
                )
            )
            session.add(
                UploadedDataset(
                    id="dataset-colleague",
                    name="Colleague upload",
                    dataset_type="herd_profile",
                    format_detected="icar_test_day",
                    user_id=4,
                    organization_id=1,
                    original_filename="colleague.csv",
                    row_count=1,
                    cow_count=1,
                )
            )
            await session.commit()
            break

    asyncio.run(_seed())
    multi_org_auth.as_user("org1_owner")
    challenges = client.get("/benchmark/challenges?organization_id=1").json()
    assert {challenge["name"] for challenge in challenges} == {"mine", "colleague"}
    assert {challenge["user_name"] for challenge in challenges} == {"Test User", "Colleague User"}
    assert [
        challenge["name"]
        for challenge in client.get("/benchmark/challenges?organization_id=1&scope=mine").json()
    ] == ["mine"]
    assert [
        challenge["name"]
        for challenge in client.get("/benchmark/challenges?organization_id=1&user_id=4").json()
    ] == ["colleague"]
    assert [
        profile["name"]
        for profile in client.get("/herd-profiles/?organization_id=1&user_id=4").json()
    ] == ["Colleague profile"]
    assert [
        dataset["name"]
        for dataset in client.get("/uploaded-datasets?organization_id=1&user_id=4").json()
    ] == ["Colleague upload"]

    async def override_colleague():
        return CurrentUser(
            id=4,
            entra_tenant_id="test-tenant",
            entra_oid="colleague-oid",
            account_type="entra",
            email="colleague@example.test",
            name="Colleague User",
            roles=["User"],
            organizations=[
                multi_org_auth.users["org1_owner"].organizations[0],
            ],
        )

    client.app.dependency_overrides[require_auth] = override_colleague
    members = client.get("/organizations/1/members")
    assert members.status_code == 200
    assert {member["user_id"] for member in members.json()} == {1, 4}
    assert [
        challenge["name"]
        for challenge in client.get("/benchmark/challenges?organization_id=1&scope=mine").json()
    ] == ["colleague"]


def test_create_organization_makes_current_user_owner(client):
    response = client.post("/organizations", json={"name": "New Dairy"})

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "New Dairy"
    assert body["role"] == "Owner"

    memberships = client.get(f"/organizations/{body['id']}/members")
    assert memberships.status_code == 200
    assert memberships.json()[0]["role"] == "Owner"


def test_invite_accept_is_idempotent_and_adds_member(client):
    created = client.post("/organizations/1/invites")
    assert created.status_code == 201
    token = created.json()["token"]

    async def invited_user():
        return CurrentUser(
            id=2,
            entra_tenant_id="other-tenant",
            entra_oid="invited-oid",
            account_type="entra",
            email="invited@example.test",
            name="Invited User",
            roles=["User"],
            organizations=[],
        )

    override = client.app.dependency_overrides[get_session]

    async def _seed_invited_user() -> None:
        async for session in override():
            session.add(
                User(
                    id=2,
                    entra_tenant_id="other-tenant",
                    entra_oid="invited-oid",
                    account_type="entra",
                    email="invited@example.test",
                    name="Invited User",
                )
            )
            await session.commit()
            break

    asyncio.run(_seed_invited_user())
    client.app.dependency_overrides[require_auth] = invited_user

    first = client.post(f"/invites/{token}/accept")
    second = client.post(f"/invites/{token}/accept")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["role"] == "Member"

    async def _counts() -> tuple[int, int]:
        async for session in override():
            memberships = await session.execute(
                select(OrganizationMembership).where(OrganizationMembership.user_id == 2)
            )
            invites = await session.execute(select(OrganizationInvite))
            return len(memberships.scalars().all()), invites.scalars().one().accepted_count
        raise AssertionError("session override did not yield")

    membership_count, accepted_count = asyncio.run(_counts())
    assert membership_count == 1
    assert accepted_count == 1


class _ScalarResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _InvitePreviewSession:
    def __init__(self, invite=None, organization=None) -> None:
        self.invite = invite
        self.organization = organization

    async def execute(self, _statement):
        return _ScalarResult(self.invite)

    async def get(self, _model, _id):
        return self.organization


def test_invite_preview_returns_organization_context():
    token = "preview-token"
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    session = _InvitePreviewSession(
        invite=OrganizationInvite(
            organization_id=1,
            token_hash=_token_hash(token),
            created_by_user_id=1,
            expires_at=expires_at,
        ),
        organization=Organization(id=1, name="Test Organization"),
    )

    body = asyncio.run(preview_invite(token, cast(AsyncSession, session)))

    assert body.organization_id == 1
    assert body.organization_name == "Test Organization"
    assert body.role == "Member"
    assert body.expires_at == expires_at


def test_invite_preview_rejects_invalid_tokens():
    session = _InvitePreviewSession()

    try:
        asyncio.run(preview_invite("not-a-real-token", cast(AsyncSession, session)))
    except HTTPException as exc:
        assert exc.status_code == 404
        assert exc.detail == "Invite is expired, revoked, or invalid."
    else:
        raise AssertionError("Expected invalid invite preview to raise HTTPException")
