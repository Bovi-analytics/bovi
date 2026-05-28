"""add challenge dataset metadata columns

Revision ID: 0010
Revises: 0009
Create Date: 2026-05-27
"""

import sqlalchemy as sa
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("challenges") as batch_op:
        batch_op.add_column(sa.Column("dataset_sources", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("dataset_stats", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("challenges") as batch_op:
        batch_op.drop_column("dataset_stats")
        batch_op.drop_column("dataset_sources")
