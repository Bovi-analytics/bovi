"""add users and organizations

Revision ID: 0006
Revises: 0005
Create Date: 2026-05-19
"""

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("entra_oid", sa.String(), nullable=False),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("role", sa.String(), nullable=False, server_default="User"),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("entra_oid"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=False)
    op.create_index(op.f("ix_users_entra_oid"), "users", ["entra_oid"], unique=True)
    op.create_index(op.f("ix_users_role"), "users", ["role"], unique=False)

    op.create_table(
        "organizations",
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "organization_memberships",
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(), nullable=False, server_default="Member"),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "organization_id", name="uq_organization_membership"),
    )
    op.create_index(
        op.f("ix_organization_memberships_organization_id"),
        "organization_memberships",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_organization_memberships_role"),
        "organization_memberships",
        ["role"],
        unique=False,
    )
    op.create_index(
        op.f("ix_organization_memberships_user_id"),
        "organization_memberships",
        ["user_id"],
        unique=False,
    )

    with op.batch_alter_table("challenges") as batch_op:
        batch_op.alter_column(
            "user_id",
            existing_type=sa.String(),
            type_=sa.Integer(),
            existing_nullable=True,
        )
        batch_op.add_column(sa.Column("organization_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_challenges_organization_id_organizations",
            "organizations",
            ["organization_id"],
            ["id"],
        )

    with op.batch_alter_table("submissions") as batch_op:
        batch_op.alter_column(
            "user_id",
            existing_type=sa.String(),
            type_=sa.Integer(),
            existing_nullable=True,
        )
        batch_op.add_column(sa.Column("organization_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_submissions_organization_id_organizations",
            "organizations",
            ["organization_id"],
            ["id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("submissions") as batch_op:
        batch_op.drop_constraint("fk_submissions_organization_id_organizations", type_="foreignkey")
        batch_op.drop_column("organization_id")
        batch_op.alter_column(
            "user_id",
            existing_type=sa.Integer(),
            type_=sa.String(),
            existing_nullable=True,
        )

    with op.batch_alter_table("challenges") as batch_op:
        batch_op.drop_constraint("fk_challenges_organization_id_organizations", type_="foreignkey")
        batch_op.drop_column("organization_id")
        batch_op.alter_column(
            "user_id",
            existing_type=sa.Integer(),
            type_=sa.String(),
            existing_nullable=True,
        )

    op.drop_index(op.f("ix_organization_memberships_user_id"), "organization_memberships")
    op.drop_index(op.f("ix_organization_memberships_role"), "organization_memberships")
    op.drop_index(op.f("ix_organization_memberships_organization_id"), "organization_memberships")
    op.drop_table("organization_memberships")
    op.drop_table("organizations")
    op.drop_index(op.f("ix_users_role"), "users")
    op.drop_index(op.f("ix_users_entra_oid"), "users")
    op.drop_index(op.f("ix_users_email"), "users")
    op.drop_table("users")
