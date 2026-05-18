"""add fitting_results table

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-18
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "fitting_results",
        sa.Column("model_type", sa.String(), nullable=False),
        sa.Column("source_app", sa.String(), nullable=False),
        sa.Column("input_data", sa.JSON(), nullable=True),
        sa.Column("output_data", sa.JSON(), nullable=True),
        sa.Column("metadata_extra", sa.JSON(), nullable=True),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_fitting_results_model_type", "fitting_results", ["model_type"])
    op.create_index("ix_fitting_results_source_app", "fitting_results", ["source_app"])


def downgrade() -> None:
    op.drop_index("ix_fitting_results_source_app", table_name="fitting_results")
    op.drop_index("ix_fitting_results_model_type", table_name="fitting_results")
    op.drop_table("fitting_results")
