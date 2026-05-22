"""Authentication and organization authorization tests."""

import asyncio

from bovi_api.auth import CurrentUser, require_auth
from bovi_api.database import get_session
from bovi_api.models import (
    Challenge,
    Organization,
    OrganizationInvite,
    OrganizationMembership,
    User,
)
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


def test_protected_route_rejects_missing_token_without_override(client):
    client.app.dependency_overrides.pop(require_auth)

    response = client.get("/benchmark/challenges")

    assert response.status_code == 401


def test_user_only_sees_challenges_for_their_organization(client):
    override = client.app.dependency_overrides[get_session]

    async def _seed() -> None:
        async for session in override():
            session.add(Organization(id=2, name="Other Organization"))
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
