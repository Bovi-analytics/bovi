"""Authentication and authorization helpers for Microsoft Entra ID."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Annotated, Any, cast

import jwt
from fastapi import Depends, HTTPException, Request, status
from jwt import PyJWKClient
from jwt.exceptions import InvalidTokenError
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from bovi_api.database import get_session
from bovi_api.models import Organization, OrganizationMembership, User
from bovi_api.settings import Settings, get_settings

APP_ROLE_ADMIN = "Admin"
APP_ROLE_USER = "User"
ORG_ROLE_OWNER = "Owner"
ORG_ROLE_MEMBER = "Member"
PERSONAL_MICROSOFT_TENANT_ID = "9188040d-6c67-4c5b-b112-36a304b66dad"


class AuthenticatedOrganization(BaseModel):
    """Organization visible to the authenticated user."""

    id: int
    name: str
    role: str


class CurrentUser(BaseModel):
    """Authenticated local Bovi user and authorization context."""

    id: int
    entra_tenant_id: str
    entra_oid: str
    account_type: str = "entra"
    email: str | None = None
    name: str | None = None
    roles: list[str] = [APP_ROLE_USER]
    is_admin: bool = False
    organizations: list[AuthenticatedOrganization] = []

    @property
    def organization_ids(self) -> list[int]:
        """Organization ids this user can access."""
        return [org.id for org in self.organizations]


class TokenIdentity(BaseModel):
    """Identity claims extracted from a verified Entra access token."""

    entra_oid: str
    entra_tenant_id: str
    account_type: str = "entra"
    email: str | None = None
    name: str | None = None
    roles: list[str] = [APP_ROLE_USER]


def _jwks_uri(tenant_id: str) -> str:
    return f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys"


def _issuers(tenant_id: str) -> list[str]:
    return [f"https://login.microsoftonline.com/{tenant_id}/v2.0"]


@lru_cache(maxsize=8)
def _jwks_client(tenant_id: str) -> PyJWKClient:
    return PyJWKClient(_jwks_uri(tenant_id), cache_keys=True)


def _extract_bearer_token(request: Request) -> str | None:
    authorization = request.headers.get("authorization")
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def validate_entra_token(token: str, settings: Settings) -> TokenIdentity:
    """Validate an Entra access token and return normalized identity claims."""
    if not settings.azure_ad_client_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication is not configured.",
        )

    try:
        unverified_claims = jwt.decode(token, options={"verify_signature": False})
        tenant_id = unverified_claims.get("tid")
        if not isinstance(tenant_id, str) or not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication token is missing tenant id.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        signing_key = _jwks_client(tenant_id).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=[settings.azure_ad_client_id, f"api://{settings.azure_ad_client_id}"],
            issuer=_issuers(tenant_id),
            options={"require": ["exp", "nbf", "iat", "tid", "oid"]},
        )
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    if claims.get("tid") != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token tenant mismatch.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    entra_oid = claims.get("oid")
    if not isinstance(entra_oid, str) or not entra_oid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token is missing object id.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    scopes = claims.get("scp")
    if not isinstance(scopes, str) or "access_as_user" not in scopes.split():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token is missing required scope.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    raw_roles = claims.get("roles") or []
    roles = [role for role in raw_roles if isinstance(role, str)]
    if not roles:
        roles = [APP_ROLE_USER]

    email = claims.get("preferred_username") or claims.get("email")
    name = claims.get("name")
    return TokenIdentity(
        entra_oid=entra_oid,
        entra_tenant_id=tenant_id,
        account_type="personal" if tenant_id == PERSONAL_MICROSOFT_TENANT_ID else "entra",
        email=email if isinstance(email, str) else None,
        name=name if isinstance(name, str) else None,
        roles=roles,
    )


def _default_organization_name(identity: TokenIdentity) -> str:
    if identity.email and "@" in identity.email:
        domain = identity.email.split("@", 1)[1].strip()
        if domain:
            return domain
    if identity.name:
        return f"{identity.name}'s organization"
    return "My organization"


async def _ensure_local_user(
    identity: TokenIdentity,
    session: AsyncSession,
) -> CurrentUser:
    result = await session.execute(
        select(User).where(
            User.entra_tenant_id == identity.entra_tenant_id,
            User.entra_oid == identity.entra_oid,
        )
    )
    user = result.scalar_one_or_none()
    now = datetime.now(timezone.utc)
    primary_role = APP_ROLE_ADMIN if APP_ROLE_ADMIN in identity.roles else APP_ROLE_USER

    if user is None:
        user = User(
            entra_tenant_id=identity.entra_tenant_id,
            entra_oid=identity.entra_oid,
            account_type=identity.account_type,
            email=identity.email,
            name=identity.name,
            role=primary_role,
            last_login_at=now,
        )
        session.add(user)
        try:
            await session.flush()
        except IntegrityError:
            await session.rollback()
            result = await session.execute(
                select(User).where(
                    User.entra_tenant_id == identity.entra_tenant_id,
                    User.entra_oid == identity.entra_oid,
                )
            )
            user = result.scalar_one_or_none()
            if user is None:
                raise
            user.email = identity.email
            user.name = identity.name
            user.account_type = identity.account_type
            user.role = primary_role
            user.last_login_at = now
    else:
        user.email = identity.email
        user.name = identity.name
        user.account_type = identity.account_type
        user.role = primary_role
        user.last_login_at = now

    await session.flush()
    assert user.id is not None

    memberships = await _memberships_for_user(user.id, session)
    if APP_ROLE_ADMIN in identity.roles:
        all_organizations = await session.execute(
            select(Organization).order_by(col(Organization.name))
        )
        memberships = [
            (
                organization,
                OrganizationMembership(
                    user_id=user.id, organization_id=organization.id or 0, role=APP_ROLE_ADMIN
                ),
            )
            for organization in all_organizations.scalars().all()
        ]

    await session.commit()
    return CurrentUser(
        id=user.id,
        entra_tenant_id=user.entra_tenant_id,
        entra_oid=user.entra_oid,
        account_type=user.account_type,
        email=user.email,
        name=user.name,
        roles=identity.roles,
        is_admin=APP_ROLE_ADMIN in identity.roles,
        organizations=[
            AuthenticatedOrganization(id=org.id or 0, name=org.name, role=membership.role)
            for org, membership in memberships
            if org.id is not None
        ],
    )


async def _memberships_for_user(
    user_id: int,
    session: AsyncSession,
) -> list[tuple[Organization, OrganizationMembership]]:
    result = await session.execute(
        select(Organization, OrganizationMembership)
        .join(
            OrganizationMembership,
            col(OrganizationMembership.organization_id) == col(Organization.id),
        )
        .where(OrganizationMembership.user_id == user_id)
        .order_by(col(Organization.name))
    )
    return [
        (cast(Organization, organization), cast(OrganizationMembership, membership))
        for organization, membership in result.all()
    ]


async def _ensure_auth_disabled_user(
    identity: TokenIdentity,
    session: AsyncSession,
) -> CurrentUser:
    """Return the local dev user and ensure the frontend's dev organization exists."""
    current_user = await _ensure_local_user(identity, session)
    organization = await session.get(Organization, 1)
    if organization is None:
        organization = Organization(
            id=1,
            name="Development Organization",
            created_by_user_id=current_user.id,
            source_entra_tenant_id=identity.entra_tenant_id,
            source_domain="local.test",
            source_display_name=identity.name,
        )
        session.add(organization)
        await session.flush()

    membership_result = await session.execute(
        select(OrganizationMembership).where(
            OrganizationMembership.user_id == current_user.id,
            OrganizationMembership.organization_id == 1,
        )
    )
    if membership_result.scalar_one_or_none() is None:
        session.add(
            OrganizationMembership(
                user_id=current_user.id,
                organization_id=1,
                role=ORG_ROLE_OWNER,
            )
        )
        await session.flush()

    await session.commit()
    return CurrentUser(
        **current_user.model_dump(exclude={"organizations"}),
        organizations=[
            AuthenticatedOrganization(id=1, name=organization.name, role=ORG_ROLE_OWNER)
        ],
    )


async def get_current_user(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> CurrentUser:
    """Validate the request bearer token and return the local Bovi user."""
    if settings.auth_disabled:
        identity = TokenIdentity(
            entra_tenant_id="dev-tenant",
            entra_oid="dev-auth-disabled",
            account_type="entra",
            email="dev@local.test",
            name="Development User",
            roles=[APP_ROLE_ADMIN],
        )
        return await _ensure_auth_disabled_user(identity, session)

    token = _extract_bearer_token(request)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    identity = validate_entra_token(token, settings)
    return await _ensure_local_user(identity, session)


require_auth = get_current_user


def require_roles(allowed_roles: set[str]):
    """Create a dependency that requires one of the provided global roles."""

    async def _dependency(
        current_user: Annotated[CurrentUser, Depends(require_auth)],
    ) -> CurrentUser:
        if not allowed_roles.intersection(current_user.roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions.",
            )
        return current_user

    return _dependency


require_admin = require_roles({APP_ROLE_ADMIN})


def user_can_access_organization(current_user: CurrentUser, organization_id: int | None) -> bool:
    """Return whether a user can access an organization-scoped record."""
    if current_user.is_admin:
        return True
    return organization_id is not None and organization_id in current_user.organization_ids


def ensure_organization_access(current_user: CurrentUser, organization_id: int | None) -> None:
    """Raise 404 when a user cannot access an organization-scoped record."""
    if not user_can_access_organization(current_user, organization_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found")


def first_accessible_organization_id(current_user: CurrentUser) -> int:
    """Return the user's default organization id."""
    if not current_user.organization_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is not a member of an organization.",
        )
    return current_user.organization_ids[0]


async def current_user_payload(current_user: CurrentUser) -> dict[str, Any]:
    """Serialize current user for auth endpoints."""
    return current_user.model_dump()
