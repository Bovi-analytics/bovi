"""add access role audits

Revision ID: 0014
Revises: 0013
Create Date: 2026-06-01
"""

import sqlalchemy as sa
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "access_role_audits",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("actor_user_id", sa.Integer(), nullable=True),
        sa.Column("target_user_id", sa.Integer(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=True),
        sa.Column("scope", sa.String(), nullable=False),
        sa.Column("old_role", sa.String(length=64), nullable=True),
        sa.Column("new_role", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["actor_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["target_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_access_role_audits_actor_user_id"), "access_role_audits", ["actor_user_id"]
    )
    op.create_index(
        op.f("ix_access_role_audits_target_user_id"), "access_role_audits", ["target_user_id"]
    )
    op.create_index(
        op.f("ix_access_role_audits_organization_id"), "access_role_audits", ["organization_id"]
    )
    op.create_index(op.f("ix_access_role_audits_scope"), "access_role_audits", ["scope"])


def downgrade() -> None:
    op.drop_index(op.f("ix_access_role_audits_scope"), table_name="access_role_audits")
    op.drop_index(op.f("ix_access_role_audits_organization_id"), table_name="access_role_audits")
    op.drop_index(op.f("ix_access_role_audits_target_user_id"), table_name="access_role_audits")
    op.drop_index(op.f("ix_access_role_audits_actor_user_id"), table_name="access_role_audits")
    op.drop_table("access_role_audits")
