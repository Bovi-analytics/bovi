"""add upload audits table

Revision ID: 0008
Revises: 0007
Create Date: 2026-05-21
"""

import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "upload_audits",
        sa.Column("action_type", sa.String(length=80), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("user_email", sa.String(length=320), nullable=True),
        sa.Column("user_name", sa.String(length=200), nullable=True),
        sa.Column("organization_id", sa.Integer(), nullable=True),
        sa.Column("organization_name", sa.String(length=200), nullable=True),
        sa.Column("original_filename", sa.String(length=500), nullable=False),
        sa.Column("content_type", sa.String(length=200), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("blob_container", sa.String(length=200), nullable=False),
        sa.Column("blob_path", sa.String(length=1000), nullable=False),
        sa.Column("challenge_id", sa.Integer(), nullable=True),
        sa.Column("submission_id", sa.Integer(), nullable=True),
        sa.Column("error_detail", sa.String(), nullable=True),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["challenge_id"], ["challenges.id"]),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["submission_id"], ["submissions.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_upload_audits_action_type", "upload_audits", ["action_type"])
    op.create_index("ix_upload_audits_status", "upload_audits", ["status"])


def downgrade() -> None:
    op.drop_index("ix_upload_audits_status", table_name="upload_audits")
    op.drop_index("ix_upload_audits_action_type", table_name="upload_audits")
    op.drop_table("upload_audits")
