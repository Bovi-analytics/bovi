"""add terms acceptance audits

Revision ID: 0017
Revises: 0016
Create Date: 2026-07-29
"""

import sqlalchemy as sa
from alembic import op

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "terms_acceptance_audits",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("terms_key", sa.String(length=128), nullable=False),
        sa.Column("terms_version", sa.String(length=64), nullable=False),
        sa.Column("document_sha256", sa.String(length=64), nullable=False),
        sa.Column("document_filename", sa.String(length=255), nullable=False),
        sa.Column("ip_address", sa.String(length=128), nullable=True),
        sa.Column("user_agent", sa.String(length=1000), nullable=True),
        sa.Column("accepted_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "terms_key",
            "terms_version",
            "document_sha256",
            name="uq_terms_acceptance_user_document",
        ),
    )
    op.create_index(
        op.f("ix_terms_acceptance_audits_user_id"),
        "terms_acceptance_audits",
        ["user_id"],
    )
    op.create_index(
        op.f("ix_terms_acceptance_audits_terms_key"),
        "terms_acceptance_audits",
        ["terms_key"],
    )
    op.create_index(
        op.f("ix_terms_acceptance_audits_terms_version"),
        "terms_acceptance_audits",
        ["terms_version"],
    )
    op.create_index(
        op.f("ix_terms_acceptance_audits_document_sha256"),
        "terms_acceptance_audits",
        ["document_sha256"],
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_terms_acceptance_audits_document_sha256"),
        table_name="terms_acceptance_audits",
    )
    op.drop_index(
        op.f("ix_terms_acceptance_audits_terms_version"),
        table_name="terms_acceptance_audits",
    )
    op.drop_index(
        op.f("ix_terms_acceptance_audits_terms_key"),
        table_name="terms_acceptance_audits",
    )
    op.drop_index(op.f("ix_terms_acceptance_audits_user_id"), table_name="terms_acceptance_audits")
    op.drop_table("terms_acceptance_audits")
