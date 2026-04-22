"""add herd_profiles table

Revision ID: 0001
Revises:
Create Date: 2026-04-15
"""

from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "herd_profiles",
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.String(length=500), nullable=False, server_default=""),
        sa.Column("achieved_21_milk", sa.Float(), nullable=False),
        sa.Column("achieved_305_milk", sa.Float(), nullable=False),
        sa.Column("achieved_75_milk", sa.Float(), nullable=False),
        sa.Column("achieved_milk", sa.Float(), nullable=False),
        sa.Column("days_dry", sa.Float(), nullable=False),
        sa.Column("days_in_milk", sa.Float(), nullable=False),
        sa.Column("days_open", sa.Float(), nullable=False),
        sa.Column("days_pregnant", sa.Float(), nullable=False),
        sa.Column("historic_calving_interval", sa.Float(), nullable=False),
        sa.Column("quality_sequence", sa.Float(), nullable=False),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_herd_profile_name"),
    )


def downgrade() -> None:
    op.drop_table("herd_profiles")
