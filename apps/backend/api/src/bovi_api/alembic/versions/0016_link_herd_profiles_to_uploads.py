"""link herd profiles to uploaded datasets

Revision ID: 0016
Revises: 0015
Create Date: 2026-06-02
"""

import sqlalchemy as sa
from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "herd_profiles",
        sa.Column("source_uploaded_dataset_id", sa.String(), nullable=True),
    )
    op.create_index(
        op.f("ix_herd_profiles_source_uploaded_dataset_id"),
        "herd_profiles",
        ["source_uploaded_dataset_id"],
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_herd_profiles_source_uploaded_dataset_id"),
        table_name="herd_profiles",
    )
    op.drop_column("herd_profiles", "source_uploaded_dataset_id")
