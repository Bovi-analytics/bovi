"""Tests for the database access management script."""

import asyncio
import importlib.util
from pathlib import Path

from bovi_api.database import create_tables, dispose_engine
from bovi_api.models import AccessRoleAudit, Organization, OrganizationMembership, User
from bovi_api.settings import get_settings
from sqlmodel import select

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "manage_access.py"
SPEC = importlib.util.spec_from_file_location("manage_access", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
manage_access = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manage_access)


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
