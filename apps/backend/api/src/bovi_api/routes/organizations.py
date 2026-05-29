"""Organization, membership, and invite endpoints."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from bovi_api.auth import (
    APP_ROLE_ADMIN,
    ORG_ROLE_MEMBER,
    ORG_ROLE_OWNER,
    CurrentUser,
    require_auth,
)
from bovi_api.database import get_session
from bovi_api.models import Organization, OrganizationInvite, OrganizationMembership, User

router = APIRouter(tags=["organizations"])


class OrganizationCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class OrganizationUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class OrganizationRead(BaseModel):
    id: int
    name: str
    role: str | None = None
    created_by_user_id: int | None = None
    source_entra_tenant_id: str | None = None
    source_domain: str | None = None
    source_display_name: str | None = None


class MemberRead(BaseModel):
    user_id: int
    email: str | None
    name: str | None
    role: str


class InviteRead(BaseModel):
    id: int
    organization_id: int
    created_by_user_id: int | None
    created_at: datetime | None
    expires_at: datetime
    revoked_at: datetime | None
    accepted_count: int
    last_accepted_at: datetime | None


class InviteCreateResponse(InviteRead):
    token: str


class InvitePreviewRead(BaseModel):
    organization_id: int
    organization_name: str
    role: str
    expires_at: datetime


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _source_domain(email: str | None) -> str | None:
    if not email or "@" not in email:
        return None
    return email.split("@", 1)[1].strip().lower() or None


async def _membership(
    session: AsyncSession, user_id: int, organization_id: int
) -> OrganizationMembership | None:
    result = await session.execute(
        select(OrganizationMembership).where(
            OrganizationMembership.user_id == user_id,
            OrganizationMembership.organization_id == organization_id,
        )
    )
    return result.scalar_one_or_none()


async def _require_owner_or_admin(
    session: AsyncSession, current_user: CurrentUser, organization_id: int
) -> None:
    if current_user.is_admin:
        return
    membership = await _membership(session, current_user.id, organization_id)
    if membership is None or membership.role != ORG_ROLE_OWNER:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Owner access required.")


async def _require_member_or_admin(
    session: AsyncSession, current_user: CurrentUser, organization_id: int
) -> None:
    if current_user.is_admin:
        return
    membership = await _membership(session, current_user.id, organization_id)
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found")


def _org_read(org: Organization, role: str | None = None) -> OrganizationRead:
    return OrganizationRead(
        id=org.id or 0,
        name=org.name,
        role=role,
        created_by_user_id=org.created_by_user_id,
        source_entra_tenant_id=org.source_entra_tenant_id,
        source_domain=org.source_domain,
        source_display_name=org.source_display_name,
    )


@router.get("/organizations", response_model=list[OrganizationRead])
async def list_organizations(
    current_user: Annotated[CurrentUser, Depends(require_auth)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[OrganizationRead]:
    """List organizations visible to the current user."""
    if current_user.is_admin:
        result = await session.execute(select(Organization).order_by(col(Organization.name)))
        return [_org_read(org, APP_ROLE_ADMIN) for org in result.scalars().all()]

    result = await session.execute(
        select(Organization, OrganizationMembership)
        .join(
            OrganizationMembership,
            col(OrganizationMembership.organization_id) == col(Organization.id),
        )
        .where(OrganizationMembership.user_id == current_user.id)
        .order_by(col(Organization.name))
    )
    return [_org_read(org, membership.role) for org, membership in result.all()]


@router.post("/organizations", response_model=OrganizationRead, status_code=201)
async def create_organization(
    body: OrganizationCreate,
    current_user: Annotated[CurrentUser, Depends(require_auth)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> OrganizationRead:
    """Create a Bovi organization and make the current user Owner."""
    org = Organization(
        name=body.name.strip(),
        created_by_user_id=current_user.id,
        source_entra_tenant_id=current_user.entra_tenant_id,
        source_domain=_source_domain(current_user.email),
        source_display_name=current_user.name,
    )
    session.add(org)
    await session.flush()
    assert org.id is not None
    session.add(
        OrganizationMembership(
            user_id=current_user.id,
            organization_id=org.id,
            role=ORG_ROLE_OWNER,
        )
    )
    await session.commit()
    await session.refresh(org)
    return _org_read(org, ORG_ROLE_OWNER)


@router.patch("/organizations/{organization_id}", response_model=OrganizationRead)
async def update_organization(
    organization_id: int,
    body: OrganizationUpdate,
    current_user: Annotated[CurrentUser, Depends(require_auth)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> OrganizationRead:
    """Rename an organization."""
    await _require_owner_or_admin(session, current_user, organization_id)
    org = await session.get(Organization, organization_id)
    if org is None:
        raise HTTPException(status_code=404, detail="Organization not found")
    org.name = body.name.strip()
    session.add(org)
    await session.commit()
    await session.refresh(org)
    return _org_read(org, APP_ROLE_ADMIN if current_user.is_admin else ORG_ROLE_OWNER)


@router.get("/organizations/{organization_id}/members", response_model=list[MemberRead])
async def list_members(
    organization_id: int,
    current_user: Annotated[CurrentUser, Depends(require_auth)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[MemberRead]:
    """List members in an organization."""
    await _require_member_or_admin(session, current_user, organization_id)
    result = await session.execute(
        select(User, OrganizationMembership)
        .join(OrganizationMembership, col(OrganizationMembership.user_id) == col(User.id))
        .where(OrganizationMembership.organization_id == organization_id)
        .order_by(col(User.email))
    )
    return [
        MemberRead(user_id=user.id or 0, email=user.email, name=user.name, role=membership.role)
        for user, membership in result.all()
    ]


@router.delete("/organizations/{organization_id}/members/{user_id}", status_code=204)
async def remove_member(
    organization_id: int,
    user_id: int,
    current_user: Annotated[CurrentUser, Depends(require_auth)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> None:
    """Remove a user's access to an organization."""
    await _require_owner_or_admin(session, current_user, organization_id)
    membership = await _membership(session, user_id, organization_id)
    if membership is None:
        raise HTTPException(status_code=404, detail="Membership not found")
    await session.delete(membership)
    await session.commit()


@router.post(
    "/organizations/{organization_id}/invites",
    response_model=InviteCreateResponse,
    status_code=201,
)
async def create_invite(
    organization_id: int,
    current_user: Annotated[CurrentUser, Depends(require_auth)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> InviteCreateResponse:
    """Create a reusable invite token. The raw token is returned only once."""
    await _require_owner_or_admin(session, current_user, organization_id)
    token = secrets.token_urlsafe(32)
    invite = OrganizationInvite(
        organization_id=organization_id,
        token_hash=_token_hash(token),
        created_by_user_id=current_user.id,
        expires_at=_now() + timedelta(days=30),
    )
    session.add(invite)
    await session.commit()
    await session.refresh(invite)
    return InviteCreateResponse(
        id=invite.id or 0,
        organization_id=invite.organization_id,
        created_by_user_id=invite.created_by_user_id,
        created_at=invite.created_at,
        expires_at=invite.expires_at,
        revoked_at=invite.revoked_at,
        accepted_count=invite.accepted_count,
        last_accepted_at=invite.last_accepted_at,
        token=token,
    )


@router.get("/organizations/{organization_id}/invites", response_model=list[InviteRead])
async def list_invites(
    organization_id: int,
    current_user: Annotated[CurrentUser, Depends(require_auth)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[OrganizationInvite]:
    """List active invite links for an organization."""
    await _require_owner_or_admin(session, current_user, organization_id)
    result = await session.execute(
        select(OrganizationInvite)
        .where(
            OrganizationInvite.organization_id == organization_id,
            col(OrganizationInvite.revoked_at).is_(None),
            col(OrganizationInvite.expires_at) > _now().replace(tzinfo=None),
        )
        .order_by(col(OrganizationInvite.created_at).desc())
    )
    return list(result.scalars().all())


@router.delete("/organizations/{organization_id}/invites/{invite_id}", status_code=204)
async def revoke_invite(
    organization_id: int,
    invite_id: int,
    current_user: Annotated[CurrentUser, Depends(require_auth)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> None:
    """Revoke an invite link."""
    await _require_owner_or_admin(session, current_user, organization_id)
    invite = await session.get(OrganizationInvite, invite_id)
    if invite is None or invite.organization_id != organization_id:
        raise HTTPException(status_code=404, detail="Invite not found")
    invite.revoked_at = _now()
    session.add(invite)
    await session.commit()


@router.get("/invites/{token}/preview", response_model=InvitePreviewRead)
async def preview_invite(
    token: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> InvitePreviewRead:
    """Return public display metadata for a valid invite token."""
    result = await session.execute(
        select(OrganizationInvite).where(OrganizationInvite.token_hash == _token_hash(token))
    )
    invite = result.scalar_one_or_none()
    now = _now()
    if invite is None or invite.revoked_at is not None or _aware(invite.expires_at) <= now:
        raise HTTPException(status_code=404, detail="Invite is expired, revoked, or invalid.")

    org = await session.get(Organization, invite.organization_id)
    if org is None:
        raise HTTPException(status_code=404, detail="Organization not found")

    return InvitePreviewRead(
        organization_id=org.id or 0,
        organization_name=org.name,
        role=ORG_ROLE_MEMBER,
        expires_at=invite.expires_at,
    )


@router.post("/invites/{token}/accept", response_model=OrganizationRead)
async def accept_invite(
    token: str,
    current_user: Annotated[CurrentUser, Depends(require_auth)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> OrganizationRead:
    """Accept an invite and join the organization as Member."""
    result = await session.execute(
        select(OrganizationInvite).where(OrganizationInvite.token_hash == _token_hash(token))
    )
    invite = result.scalar_one_or_none()
    now = _now()
    if invite is None or invite.revoked_at is not None or _aware(invite.expires_at) <= now:
        raise HTTPException(status_code=404, detail="Invite is expired, revoked, or invalid.")

    org = await session.get(Organization, invite.organization_id)
    if org is None:
        raise HTTPException(status_code=404, detail="Organization not found")

    membership = await _membership(session, current_user.id, invite.organization_id)
    if membership is None:
        membership = OrganizationMembership(
            user_id=current_user.id,
            organization_id=invite.organization_id,
            role=ORG_ROLE_MEMBER,
        )
        session.add(membership)
        invite.accepted_count += 1
        invite.last_accepted_at = now
        session.add(invite)
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
    else:
        await session.commit()
    return _org_read(org, membership.role)
