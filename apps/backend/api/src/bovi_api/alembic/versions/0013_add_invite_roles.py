"""add invite roles

Revision ID: 0013
Revises: 0012
Create Date: 2026-05-29
"""

import sqlalchemy as sa
from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("organization_invites") as batch_op:
        batch_op.add_column(sa.Column("role", sa.String(), nullable=False, server_default="Member"))
    op.create_index(op.f("ix_organization_invites_role"), "organization_invites", ["role"])


def downgrade() -> None:
    op.drop_index(op.f("ix_organization_invites_role"), table_name="organization_invites")
    with op.batch_alter_table("organization_invites") as batch_op:
        batch_op.drop_column("role")
