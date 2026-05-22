"""Token validation tests for multi-tenant Microsoft auth."""

import bovi_api.auth as auth
import pytest
from bovi_api.auth import PERSONAL_MICROSOFT_TENANT_ID, validate_entra_token
from bovi_api.settings import Settings
from fastapi import HTTPException


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
