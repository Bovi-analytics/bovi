"""Tests for the database access management script."""

import asyncio

import bovi_api.auth as auth
from bovi_api.auth import PENDING_BOOTSTRAP_TENANT_ID, TokenIdentity, pending_bootstrap_oid
from bovi_api.database import create_tables, dispose_engine
from bovi_api.models import AccessRoleAudit, Organization, OrganizationMembership, User
from bovi_api.scripts import manage_access
from bovi_api.settings import get_settings
from sqlmodel import select


def test_manage_access_grants_admin_to_existing_user(tmp_path, monkeypatch):
    async def _run() -> None:
        monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'access.db'}")
        get_settings.cache_clear()
        await dispose_engine()
        try:
            await create_tables()
            factory = manage_access._get_session_factory()
            async with factory() as session:
                session.add(
                    User(
                        entra_tenant_id="tenant-a",
                        entra_oid="user-oid",
                        account_type="entra",
                        email="user@example.test",
                        name="Script User",
                    )
                )
                await session.commit()

            await manage_access._main(["grant-admin", "--email", "user@example.test"])

            async with factory() as session:
                user = (
                    await session.execute(select(User).where(User.email == "user@example.test"))
                ).scalar_one()
                audit = (
                    await session.execute(
                        select(AccessRoleAudit).where(AccessRoleAudit.target_user_id == user.id)
                    )
                ).scalar_one()
                assert user.role == "Admin"
                assert audit.old_role == "User"
                assert audit.new_role == "Admin"
        finally:
            await dispose_engine()
            get_settings.cache_clear()

    asyncio.run(_run())


def test_manage_access_bootstraps_default_admin_as_pending_user(tmp_path, monkeypatch):
    async def _run() -> None:
        monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'access.db'}")
        monkeypatch.delenv("BOVI_BOOTSTRAP_ADMIN_EMAILS", raising=False)
        get_settings.cache_clear()
        await dispose_engine()
        try:
            await create_tables()
            await manage_access._main(["bootstrap-admins"])

            factory = manage_access._get_session_factory()
            async with factory() as session:
                user = (
                    await session.execute(select(User).where(User.email == "douwedekok@gmail.com"))
                ).scalar_one()
                assert user.entra_tenant_id == PENDING_BOOTSTRAP_TENANT_ID
                assert user.entra_oid == pending_bootstrap_oid("douwedekok@gmail.com")
                assert user.role == "Admin"
        finally:
            await dispose_engine()
            get_settings.cache_clear()

    asyncio.run(_run())


def test_pending_bootstrap_admin_is_claimed_on_first_login(tmp_path, monkeypatch):
    async def _run() -> None:
        monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'access.db'}")
        get_settings.cache_clear()
        await dispose_engine()
        try:
            await create_tables()
            await manage_access._main(["grant-admin", "--email", "douwedekok@gmail.com"])

            factory = manage_access._get_session_factory()
            async with factory() as session:
                current_user = await auth._ensure_local_user(
                    TokenIdentity(
                        entra_tenant_id="real-tenant",
                        entra_oid="real-oid",
                        account_type="entra",
                        email="douwedekok@gmail.com",
                        name="Douwe de Kok",
                    ),
                    session,
                )
                assert current_user.is_admin is True
                assert current_user.roles == ["Admin"]

            async with factory() as session:
                users = (await session.execute(select(User))).scalars().all()
                assert len(users) == 1
                assert users[0].entra_tenant_id == "real-tenant"
                assert users[0].entra_oid == "real-oid"
                assert users[0].role == "Admin"
        finally:
            await dispose_engine()
            get_settings.cache_clear()

    asyncio.run(_run())


def test_manage_access_sets_organization_role_idempotently(tmp_path, monkeypatch):
    async def _run() -> None:
        monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'access.db'}")
        get_settings.cache_clear()
        await dispose_engine()
        try:
            await create_tables()
            factory = manage_access._get_session_factory()
            async with factory() as session:
                session.add(
                    User(
                        entra_tenant_id="tenant-a",
                        entra_oid="user-oid",
                        account_type="entra",
                        email="user@example.test",
                        name="Script User",
                    )
                )
                session.add(Organization(id=5, name="Acme Dairy"))
                await session.commit()

            await manage_access._main(
                [
                    "set-org-role",
                    "--email",
                    "user@example.test",
                    "--organization-id",
                    "5",
                    "--role",
                    "Owner",
                ]
            )
            await manage_access._main(
                [
                    "set-org-role",
                    "--email",
                    "user@example.test",
                    "--organization-id",
                    "5",
                    "--role",
                    "Owner",
                ]
            )

            async with factory() as session:
                memberships = (
                    (await session.execute(select(OrganizationMembership))).scalars().all()
                )
                audits = (await session.execute(select(AccessRoleAudit))).scalars().all()
                assert len(memberships) == 1
                assert memberships[0].role == "Owner"
                assert len(audits) == 1
        finally:
            await dispose_engine()
            get_settings.cache_clear()

    asyncio.run(_run())
