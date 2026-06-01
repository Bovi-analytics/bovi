"""Token validation tests for multi-tenant Microsoft auth."""

import asyncio
from typing import cast

import bovi_api.auth as auth
import pytest
from bovi_api.auth import PERSONAL_MICROSOFT_TENANT_ID, TokenIdentity, validate_entra_token
from bovi_api.models import Organization, OrganizationMembership, User
from bovi_api.settings import Settings
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import select
from starlette.requests import Request


class _SigningKey:
    key = "public-key"


class _JwksClient:
    def get_signing_key_from_jwt(self, token: str) -> _SigningKey:
        assert token == "token"
        return _SigningKey()


def _patch_decode(monkeypatch, claims: dict) -> None:
    def fake_decode(token, key="", algorithms=None, audience=None, issuer=None, options=None):
        if key == "":
            return {"tid": claims.get("tid")}
        assert audience == ["client-id", "api://client-id"]
        assert issuer == [f"https://login.microsoftonline.com/{claims['tid']}/v2.0"]
        return claims

    monkeypatch.setattr(auth, "_jwks_client", lambda tenant_id: _JwksClient())
    monkeypatch.setattr(auth.jwt, "decode", fake_decode)


def test_validate_entra_token_accepts_work_account(monkeypatch):
    _patch_decode(
        monkeypatch,
        {
            "tid": "tenant-a",
            "oid": "object-a",
            "scp": "access_as_user",
            "preferred_username": "user@example.com",
            "name": "User",
            "exp": 1,
            "nbf": 1,
            "iat": 1,
        },
    )

    identity = validate_entra_token("token", Settings(azure_ad_client_id="client-id"))

    assert identity.entra_tenant_id == "tenant-a"
    assert identity.entra_oid == "object-a"
    assert identity.account_type == "entra"


def test_validate_entra_token_marks_personal_account(monkeypatch):
    _patch_decode(
        monkeypatch,
        {
            "tid": PERSONAL_MICROSOFT_TENANT_ID,
            "oid": "personal-object",
            "scp": "access_as_user",
            "exp": 1,
            "nbf": 1,
            "iat": 1,
        },
    )

    identity = validate_entra_token("token", Settings(azure_ad_client_id="client-id"))

    assert identity.account_type == "personal"


def test_get_current_user_rejects_missing_bearer_token():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            auth.get_current_user(
                request=Request({"type": "http", "headers": []}),
                session=cast(AsyncSession, None),
                settings=Settings(auth_disabled=False),
            )
        )

    assert exc.value.status_code == 401
    assert exc.value.detail == "Authentication required."


def test_auth_disabled_seeds_development_organization():
    async def _run() -> None:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(User.__table__.create)  # type: ignore[union-attr]
                await conn.run_sync(Organization.__table__.create)  # type: ignore[union-attr]
                await conn.run_sync(OrganizationMembership.__table__.create)  # type: ignore[union-attr]

            async with session_factory() as session:
                current_user = await auth.get_current_user(
                    request=Request({"type": "http", "headers": []}),
                    session=session,
                    settings=Settings(auth_disabled=True),
                )

                assert current_user.organizations[0].id == 1
                assert current_user.organizations[0].name == "Development Organization"
                assert current_user.organizations[0].role == "Owner"
                assert current_user.is_admin is True

            async with session_factory() as session:
                organization = await session.get(Organization, 1)
                membership_result = await session.execute(
                    select(OrganizationMembership).where(
                        OrganizationMembership.user_id == 1,
                        OrganizationMembership.organization_id == 1,
                    )
                )

                assert organization is not None
                assert (await session.get(User, 1)).role == "Admin"  # type: ignore[union-attr]
                assert membership_result.scalar_one_or_none() is not None
        finally:
            await engine.dispose()

    asyncio.run(_run())


@pytest.mark.parametrize(
    "claims",
    [
        {"tid": "tenant-a", "scp": "access_as_user"},
        {"tid": "tenant-a", "oid": "object-a", "scp": "wrong_scope"},
        {"oid": "object-a", "scp": "access_as_user"},
    ],
)
def test_validate_entra_token_rejects_missing_required_claims(monkeypatch, claims):
    _patch_decode(monkeypatch, claims)

    with pytest.raises(HTTPException) as exc:
        validate_entra_token("token", Settings(azure_ad_client_id="client-id"))

    assert exc.value.status_code == 401


class _ScalarResult:
    def __init__(self, value=None, rows=None) -> None:
        self.value = value
        self.rows = rows or []

    def scalar_one_or_none(self):
        return self.value

    def all(self):
        return self.rows

    def scalars(self):
        return self


class _FakeSession:
    def __init__(self, user: User) -> None:
        self.user = user
        self.execute_count = 0
        self.flush_count = 0
        self.rollback_count = 0
        self.commit_count = 0

    async def execute(self, _statement):
        self.execute_count += 1
        if self.execute_count == 1:
            return _ScalarResult()
        if self.execute_count == 2:
            return _ScalarResult(self.user)
        return _ScalarResult(rows=[])

    def add(self, _model) -> None:
        return None

    async def flush(self) -> None:
        self.flush_count += 1
        if self.flush_count == 1:
            raise IntegrityError("insert user", {}, Exception("duplicate identity"))

    async def rollback(self) -> None:
        self.rollback_count += 1

    async def commit(self) -> None:
        self.commit_count += 1


def test_ensure_local_user_recovers_from_concurrent_insert_race():
    existing_user = User(
        id=42,
        entra_tenant_id="dev-tenant",
        entra_oid="dev-auth-disabled",
        account_type="entra",
        email="old@example.test",
        name="Old Name",
        role="User",
    )
    session = _FakeSession(existing_user)

    current_user = asyncio.run(
        auth._ensure_local_user(
            TokenIdentity(
                entra_tenant_id="dev-tenant",
                entra_oid="dev-auth-disabled",
                account_type="entra",
                email="dev@local.test",
                name="Development User",
            ),
            session,  # type: ignore[arg-type]
        )
    )

    assert session.rollback_count == 1
    assert session.commit_count == 1
    assert current_user.id == 42
    assert current_user.email == "dev@local.test"
    assert current_user.is_admin is False
    assert existing_user.role == "User"


def test_ensure_local_user_uses_database_global_role_instead_of_token_roles():
    async def _run() -> None:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(User.__table__.create)  # type: ignore[union-attr]
                await conn.run_sync(Organization.__table__.create)  # type: ignore[union-attr]
                await conn.run_sync(OrganizationMembership.__table__.create)  # type: ignore[union-attr]

            async with session_factory() as session:
                admin = User(
                    entra_tenant_id="tenant-a",
                    entra_oid="admin-oid",
                    account_type="entra",
                    email="admin@example.test",
                    name="DB Admin",
                    role="Admin",
                )
                token_admin_denied = User(
                    entra_tenant_id="tenant-a",
                    entra_oid="token-admin-oid",
                    account_type="entra",
                    email="token-admin@example.test",
                    name="Token Admin",
                    role="User",
                )
                session.add(admin)
                session.add(token_admin_denied)
                await session.commit()

            async with session_factory() as session:
                current_user = await auth._ensure_local_user(
                    TokenIdentity(
                        entra_tenant_id="tenant-a",
                        entra_oid="admin-oid",
                        account_type="entra",
                        email="admin-updated@example.test",
                        name="Updated DB Admin",
                    ),
                    session,
                )
                assert current_user.is_admin is True
                assert current_user.roles == ["Admin"]

            async with session_factory() as session:
                current_user = await auth._ensure_local_user(
                    TokenIdentity(
                        entra_tenant_id="tenant-a",
                        entra_oid="token-admin-oid",
                        account_type="entra",
                        email="token-admin@example.test",
                        name="Token Admin",
                    ),
                    session,
                )
                assert current_user.is_admin is False
                assert current_user.roles == ["User"]
        finally:
            await engine.dispose()

    asyncio.run(_run())


def test_ensure_local_user_new_user_defaults_to_user_role():
    async def _run() -> None:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(User.__table__.create)  # type: ignore[union-attr]
                await conn.run_sync(Organization.__table__.create)  # type: ignore[union-attr]
                await conn.run_sync(OrganizationMembership.__table__.create)  # type: ignore[union-attr]

            async with session_factory() as session:
                current_user = await auth._ensure_local_user(
                    TokenIdentity(
                        entra_tenant_id="tenant-a",
                        entra_oid="new-user-oid",
                        account_type="entra",
                        email="new@example.test",
                        name="New User",
                    ),
                    session,
                )

                assert current_user.is_admin is False
                assert current_user.roles == ["User"]

                user = (
                    await session.execute(select(User).where(User.entra_oid == "new-user-oid"))
                ).scalar_one()
                assert user.role == "User"
        finally:
            await engine.dispose()

    asyncio.run(_run())
