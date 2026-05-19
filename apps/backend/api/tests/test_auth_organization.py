"""Authentication and organization authorization tests."""

import asyncio

from bovi_api.auth import CurrentUser, require_auth
from bovi_api.database import get_session
from bovi_api.models import Challenge, Organization


def test_auth_me_returns_current_user(client):
    response = client.get("/auth/me")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 1
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

    response = client.get("/benchmark/challenges")

    assert response.status_code == 200
    assert [challenge["name"] for challenge in response.json()] == ["visible"]


def test_admin_can_see_challenges_for_all_organizations(client):
    async def override_admin():
        return CurrentUser(
            id=7,
            entra_oid="admin-oid",
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

    response = client.get("/benchmark/challenges")

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
