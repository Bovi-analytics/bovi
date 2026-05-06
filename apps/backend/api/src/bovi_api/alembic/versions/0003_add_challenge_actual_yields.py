"""add actual_yields to challenges

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-01
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("challenges") as batch_op:
        batch_op.add_column(sa.Column("actual_yields", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("challenges") as batch_op:
        batch_op.drop_column("actual_yields")
