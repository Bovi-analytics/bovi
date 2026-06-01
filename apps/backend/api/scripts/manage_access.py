"""Manage database-backed Bovi access roles.

Examples:
    uv run python apps/backend/api/scripts/manage_access.py grant-admin --email admin@example.com
    uv run python apps/backend/api/scripts/manage_access.py set-org-role \
        --email user@example.com --organization "Acme Dairy" --role Owner
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from bovi_api.auth import APP_ROLE_ADMIN, APP_ROLE_USER, ORG_ROLE_MEMBER, ORG_ROLE_OWNER
from bovi_api.database import _get_session_factory
from bovi_api.models import AccessRoleAudit, Organization, OrganizationMembership, User
from sqlalchemy import func
from sqlmodel import select


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage Bovi access roles in the database.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    for name in ("grant-admin", "revoke-admin"):
        command = subcommands.add_parser(name)
        _add_user_args(command)

    org = subcommands.add_parser("set-org-role")
    _add_user_args(org)
    org.add_argument("--organization-id", type=int)
    org.add_argument("--organization")
    org.add_argument("--role", choices=[ORG_ROLE_OWNER, ORG_ROLE_MEMBER], required=True)

    return parser


def _add_user_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--email", required=True)
    parser.add_argument("--entra-tenant-id")
    parser.add_argument("--entra-oid")
    parser.add_argument("--name")
    parser.add_argument("--account-type", default="entra")


async def _find_or_create_user(args: argparse.Namespace) -> User:
    factory = _get_session_factory()
    async with factory() as session:
        users = (
            (await session.execute(select(User).where(User.email == args.email))).scalars().all()
        )
        if len(users) > 1:
            raise SystemExit(f"Multiple users found for email {args.email}; use a unique identity.")
        if users:
            return users[0]

        if not args.entra_tenant_id or not args.entra_oid:
            raise SystemExit(
                "User has not logged in yet. Provide --entra-tenant-id and --entra-oid "
                "to pre-create."
            )

        user = User(
            entra_tenant_id=args.entra_tenant_id,
            entra_oid=args.entra_oid,
            account_type=args.account_type,
            email=args.email,
            name=args.name,
            role=APP_ROLE_USER,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


async def _set_global_role(args: argparse.Namespace, role: str) -> None:
    user = await _find_or_create_user(args)
    factory = _get_session_factory()
    async with factory() as session:
        stored = await session.get(User, user.id)
        if stored is None:
            raise SystemExit("User disappeared while updating role.")
        old_role = stored.role
        if old_role == role:
            print(f"{stored.email} already has global role {role}.")
            return
        if old_role == APP_ROLE_ADMIN and role == APP_ROLE_USER:
            admin_count = int(
                (
                    await session.execute(
                        select(func.count()).select_from(User).where(User.role == APP_ROLE_ADMIN)
                    )
                ).scalar_one()
            )
            if admin_count <= 1:
                raise SystemExit("Cannot remove the last global admin.")
        stored.role = role
        session.add(stored)
        session.add(
            AccessRoleAudit(
                actor_user_id=None,
                target_user_id=stored.id or 0,
                scope="global",
                old_role=old_role,
                new_role=role,
            )
        )
        await session.commit()
        print(f"Set {stored.email} global role to {role}.")


async def _find_organization(args: argparse.Namespace) -> Organization:
    factory = _get_session_factory()
    async with factory() as session:
        if args.organization_id is not None:
            organization = await session.get(Organization, args.organization_id)
            if organization is None:
                raise SystemExit(f"Organization id {args.organization_id} not found.")
            return organization
        if not args.organization:
            raise SystemExit("Provide --organization-id or --organization.")
        organizations = (
            (
                await session.execute(
                    select(Organization).where(Organization.name == args.organization)
                )
            )
            .scalars()
            .all()
        )
        if len(organizations) != 1:
            raise SystemExit(f"Expected exactly one organization named {args.organization}.")
        return organizations[0]


async def _set_org_role(args: argparse.Namespace) -> None:
    user = await _find_or_create_user(args)
    organization = await _find_organization(args)
    factory = _get_session_factory()
    async with factory() as session:
        membership = (
            await session.execute(
                select(OrganizationMembership).where(
                    OrganizationMembership.user_id == user.id,
                    OrganizationMembership.organization_id == organization.id,
                )
            )
        ).scalar_one_or_none()
        old_role = membership.role if membership else None
        if old_role == ORG_ROLE_OWNER and args.role != ORG_ROLE_OWNER:
            owner_count = int(
                (
                    await session.execute(
                        select(func.count())
                        .select_from(OrganizationMembership)
                        .where(
                            OrganizationMembership.organization_id == organization.id,
                            OrganizationMembership.role == ORG_ROLE_OWNER,
                        )
                    )
                ).scalar_one()
            )
            if owner_count <= 1:
                raise SystemExit("Cannot remove the last organization owner.")
        if membership is None:
            membership = OrganizationMembership(
                user_id=user.id or 0,
                organization_id=organization.id or 0,
                role=args.role,
            )
        elif membership.role == args.role:
            print(f"{user.email} already has organization role {args.role}.")
            return
        else:
            membership.role = args.role
        session.add(membership)
        session.add(
            AccessRoleAudit(
                actor_user_id=None,
                target_user_id=user.id or 0,
                organization_id=organization.id,
                scope="organization",
                old_role=old_role,
                new_role=args.role,
            )
        )
        await session.commit()
        print(f"Set {user.email} role for {organization.name} to {args.role}.")


async def _main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "grant-admin":
        await _set_global_role(args, APP_ROLE_ADMIN)
    elif args.command == "revoke-admin":
        await _set_global_role(args, APP_ROLE_USER)
    elif args.command == "set-org-role":
        await _set_org_role(args)


if __name__ == "__main__":
    asyncio.run(_main())
