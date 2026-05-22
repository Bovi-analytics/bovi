"""multi tenant invites and herd profile scoping

Revision ID: 0009
Revises: 0008
Create Date: 2026-05-22
"""

import sqlalchemy as sa
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(
            sa.Column(
                "entra_tenant_id", sa.String(), nullable=False, server_default="legacy-tenant"
            )
        )
        batch_op.add_column(
            sa.Column("account_type", sa.String(), nullable=False, server_default="entra")
        )
        batch_op.create_unique_constraint(
            "uq_user_entra_identity", ["entra_tenant_id", "entra_oid"]
        )
    op.create_index(op.f("ix_users_entra_tenant_id"), "users", ["entra_tenant_id"])
    op.create_index(op.f("ix_users_account_type"), "users", ["account_type"])

    with op.batch_alter_table("organizations") as batch_op:
        batch_op.add_column(sa.Column("created_by_user_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("source_entra_tenant_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("source_domain", sa.String(length=320), nullable=True))
        batch_op.add_column(sa.Column("source_display_name", sa.String(length=200), nullable=True))
        batch_op.create_foreign_key(
            "fk_organizations_created_by_user_id_users",
            "users",
            ["created_by_user_id"],
            ["id"],
        )
    op.create_index(
        op.f("ix_organizations_source_entra_tenant_id"),
        "organizations",
        ["source_entra_tenant_id"],
    )

    op.create_table(
        "organization_invites",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("accepted_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index(
        op.f("ix_organization_invites_organization_id"),
        "organization_invites",
        ["organization_id"],
    )
    op.create_index(
        op.f("ix_organization_invites_token_hash"),
        "organization_invites",
        ["token_hash"],
        unique=True,
    )

    with op.batch_alter_table("herd_profiles") as batch_op:
        batch_op.add_column(sa.Column("user_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("organization_id", sa.Integer(), nullable=True))
        batch_op.drop_constraint("uq_herd_profile_name", type_="unique")
        batch_op.create_unique_constraint("uq_herd_profile_org_name", ["organization_id", "name"])
        batch_op.create_foreign_key("fk_herd_profiles_user_id_users", "users", ["user_id"], ["id"])
        batch_op.create_foreign_key(
            "fk_herd_profiles_organization_id_organizations",
            "organizations",
            ["organization_id"],
            ["id"],
        )
    op.create_index(op.f("ix_herd_profiles_user_id"), "herd_profiles", ["user_id"])
    op.create_index(op.f("ix_herd_profiles_organization_id"), "herd_profiles", ["organization_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_herd_profiles_organization_id"), table_name="herd_profiles")
    op.drop_index(op.f("ix_herd_profiles_user_id"), table_name="herd_profiles")
    with op.batch_alter_table("herd_profiles") as batch_op:
        batch_op.drop_constraint(
            "fk_herd_profiles_organization_id_organizations", type_="foreignkey"
        )
        batch_op.drop_constraint("fk_herd_profiles_user_id_users", type_="foreignkey")
        batch_op.drop_constraint("uq_herd_profile_org_name", type_="unique")
        batch_op.create_unique_constraint("uq_herd_profile_name", ["name"])
        batch_op.drop_column("organization_id")
        batch_op.drop_column("user_id")

    op.drop_index(op.f("ix_organization_invites_token_hash"), table_name="organization_invites")
    op.drop_index(
        op.f("ix_organization_invites_organization_id"), table_name="organization_invites"
    )
    op.drop_table("organization_invites")

    op.drop_index(op.f("ix_organizations_source_entra_tenant_id"), table_name="organizations")
    with op.batch_alter_table("organizations") as batch_op:
        batch_op.drop_constraint("fk_organizations_created_by_user_id_users", type_="foreignkey")
        batch_op.drop_column("source_display_name")
        batch_op.drop_column("source_domain")
        batch_op.drop_column("source_entra_tenant_id")
        batch_op.drop_column("created_by_user_id")

    op.drop_index(op.f("ix_users_account_type"), table_name="users")
    op.drop_index(op.f("ix_users_entra_tenant_id"), table_name="users")
    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_constraint("uq_user_entra_identity", type_="unique")
        batch_op.drop_column("account_type")
        batch_op.drop_column("entra_tenant_id")
