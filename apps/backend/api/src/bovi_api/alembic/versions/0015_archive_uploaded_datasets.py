"""archive uploaded datasets

Revision ID: 0015
Revises: 0014
Create Date: 2026-06-01
"""

import sqlalchemy as sa
from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "uploaded_datasets",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column("uploaded_datasets", sa.Column("deleted_by_user_id", sa.Integer(), nullable=True))
    op.create_index(
        op.f("ix_uploaded_datasets_deleted_by_user_id"),
        "uploaded_datasets",
        ["deleted_by_user_id"],
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_uploaded_datasets_deleted_by_user_id"), table_name="uploaded_datasets")
    op.drop_column("uploaded_datasets", "deleted_by_user_id")
    op.drop_column("uploaded_datasets", "deleted_at")
