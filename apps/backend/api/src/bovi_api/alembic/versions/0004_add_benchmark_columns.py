"""add benchmark_model, name, source columns

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-01
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("submissions") as batch_op:
        batch_op.add_column(sa.Column("benchmark_model", sa.String(), nullable=True))
    with op.batch_alter_table("challenges") as batch_op:
        batch_op.add_column(sa.Column("name", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("source", sa.String(), nullable=True))
        batch_op.alter_column("reference_yields", existing_type=sa.JSON(), nullable=True)


def downgrade() -> None:
    with op.batch_alter_table("submissions") as batch_op:
        batch_op.drop_column("benchmark_model")
    with op.batch_alter_table("challenges") as batch_op:
        batch_op.drop_column("name")
        batch_op.drop_column("source")
