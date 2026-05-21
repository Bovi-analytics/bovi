"""add submission run_options

Revision ID: 0006
Revises: 0005
Create Date: 2026-05-21
"""

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("submissions", sa.Column("run_options", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("submissions", "run_options")
