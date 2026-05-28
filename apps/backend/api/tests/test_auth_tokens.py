"""Token validation tests for multi-tenant Microsoft auth."""

import asyncio

import bovi_api.auth as auth
import pytest
from bovi_api.auth import PERSONAL_MICROSOFT_TENANT_ID, TokenIdentity, validate_entra_token
from bovi_api.models import User
from bovi_api.settings import Settings
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError


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
                roles=["Admin"],
            ),
            session,  # type: ignore[arg-type]
        )
    )

    assert session.rollback_count == 1
    assert session.commit_count == 1
    assert current_user.id == 42
    assert current_user.email == "dev@local.test"
    assert current_user.is_admin is True
