"""add challenges and submissions tables

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-01
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "challenges",
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("size", sa.String(), nullable=False),
        sa.Column("period", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("cow_metadata", sa.JSON(), nullable=False),
        sa.Column("reference_yields", sa.JSON(), nullable=False),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "submissions",
        sa.Column("submission_type", sa.String(), nullable=False),
        sa.Column("model_type", sa.String(), nullable=True),
        sa.Column("organization", sa.String(), nullable=True),
        sa.Column("country", sa.String(), nullable=True),
        sa.Column("calculation_method", sa.String(), nullable=True),
        sa.Column("notes", sa.String(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("challenge_id", sa.Integer(), nullable=False),
        sa.Column("submitted_yields", sa.JSON(), nullable=False),
        sa.Column("bovi_yields", sa.JSON(), nullable=False),
        sa.Column("stats", sa.JSON(), nullable=False),
        sa.Column("failed_cow_ids", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["challenge_id"], ["challenges.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("submissions")
    op.drop_table("challenges")
