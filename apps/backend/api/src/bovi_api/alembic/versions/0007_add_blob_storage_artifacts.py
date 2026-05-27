"""add blob-backed storage artifacts

Revision ID: 0007
Revises: 0006
Create Date: 2026-05-27
"""

import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "storage_artifacts",
        sa.Column("artifact_kind", sa.String(), nullable=False),
        sa.Column("entity_type", sa.String(), nullable=False),
        sa.Column("entity_uuid", sa.String(), nullable=True),
        sa.Column("storage_account_name", sa.String(), nullable=False),
        sa.Column("container_name", sa.String(), nullable=False),
        sa.Column("blob_path", sa.String(), nullable=False),
        sa.Column("original_filename", sa.String(), nullable=True),
        sa.Column("content_type", sa.String(), nullable=False),
        sa.Column("content_encoding", sa.String(), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(), nullable=False),
        sa.Column("etag", sa.String(), nullable=True),
        sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column("record_count", sa.Integer(), nullable=True),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("metadata_extra", sa.JSON(), nullable=True),
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("blob_path", name="uq_storage_artifact_blob_path"),
    )
    op.create_index("ix_storage_artifacts_artifact_kind", "storage_artifacts", ["artifact_kind"])
    op.create_index("ix_storage_artifacts_entity_type", "storage_artifacts", ["entity_type"])
    op.create_index("ix_storage_artifacts_entity_uuid", "storage_artifacts", ["entity_uuid"])
    op.create_index("ix_storage_artifacts_blob_path", "storage_artifacts", ["blob_path"])

    op.create_table(
        "uploaded_datasets",
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("dataset_type", sa.String(), nullable=False),
        sa.Column("format_detected", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("raw_file_artifact_id", sa.String(), nullable=True),
        sa.Column("cows_artifact_id", sa.String(), nullable=True),
        sa.Column("stats_artifact_id", sa.String(), nullable=True),
        sa.Column("original_filename", sa.String(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("cow_count", sa.Integer(), nullable=True),
        sa.Column("detected_parity", sa.Integer(), nullable=True),
        sa.Column("columns", sa.JSON(), nullable=True),
        sa.Column("column_mapping", sa.JSON(), nullable=True),
        sa.Column("warnings", sa.JSON(), nullable=True),
        sa.Column("stats_summary", sa.JSON(), nullable=True),
        sa.Column("raw_stats_summary", sa.JSON(), nullable=True),
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["raw_file_artifact_id"], ["storage_artifacts.id"]),
        sa.ForeignKeyConstraint(["cows_artifact_id"], ["storage_artifacts.id"]),
        sa.ForeignKeyConstraint(["stats_artifact_id"], ["storage_artifacts.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_uploaded_datasets_dataset_type", "uploaded_datasets", ["dataset_type"])
    op.create_index("ix_uploaded_datasets_user_id", "uploaded_datasets", ["user_id"])

    with op.batch_alter_table("challenges") as batch_op:
        batch_op.add_column(sa.Column("uuid", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("test_day_file_artifact_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("actual_yields_file_artifact_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("cow_metadata_artifact_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("actual_yields_artifact_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("test_day_filename", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("actual_yields_filename", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("row_count", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("cow_count", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("actual_yield_count", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("herd_count", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("parity_counts", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("column_mapping", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("ingest_warnings", sa.JSON(), nullable=True))
        batch_op.add_column(
            sa.Column("ingest_status", sa.String(), nullable=False, server_default="ready")
        )
        batch_op.create_index("ix_challenges_uuid", ["uuid"], unique=True)
        batch_op.create_index("ix_challenges_ingest_status", ["ingest_status"])
        batch_op.create_foreign_key(
            "fk_challenges_test_day_file_artifact",
            "storage_artifacts",
            ["test_day_file_artifact_id"],
            ["id"],
        )
        batch_op.create_foreign_key(
            "fk_challenges_actual_yields_file_artifact",
            "storage_artifacts",
            ["actual_yields_file_artifact_id"],
            ["id"],
        )
        batch_op.create_foreign_key(
            "fk_challenges_cow_metadata_artifact",
            "storage_artifacts",
            ["cow_metadata_artifact_id"],
            ["id"],
        )
        batch_op.create_foreign_key(
            "fk_challenges_actual_yields_artifact",
            "storage_artifacts",
            ["actual_yields_artifact_id"],
            ["id"],
        )
        batch_op.alter_column("cow_metadata", existing_type=sa.JSON(), nullable=True)

    with op.batch_alter_table("submissions") as batch_op:
        batch_op.add_column(sa.Column("uuid", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("input_file_artifact_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("submitted_yields_artifact_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("bovi_yields_artifact_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("input_filename", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("row_count", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("submitted_yield_count", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("benchmark_yield_count", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column("failed_count", sa.Integer(), nullable=False, server_default="0")
        )
        batch_op.add_column(
            sa.Column("ingest_status", sa.String(), nullable=False, server_default="ready")
        )
        batch_op.create_index("ix_submissions_uuid", ["uuid"], unique=True)
        batch_op.create_index("ix_submissions_ingest_status", ["ingest_status"])
        batch_op.create_foreign_key(
            "fk_submissions_input_file_artifact",
            "storage_artifacts",
            ["input_file_artifact_id"],
            ["id"],
        )
        batch_op.create_foreign_key(
            "fk_submissions_submitted_yields_artifact",
            "storage_artifacts",
            ["submitted_yields_artifact_id"],
            ["id"],
        )
        batch_op.create_foreign_key(
            "fk_submissions_bovi_yields_artifact",
            "storage_artifacts",
            ["bovi_yields_artifact_id"],
            ["id"],
        )
        batch_op.alter_column("submitted_yields", existing_type=sa.JSON(), nullable=True)
        batch_op.alter_column("bovi_yields", existing_type=sa.JSON(), nullable=True)


def downgrade() -> None:
    with op.batch_alter_table("submissions") as batch_op:
        batch_op.drop_constraint("fk_submissions_bovi_yields_artifact", type_="foreignkey")
        batch_op.drop_constraint("fk_submissions_submitted_yields_artifact", type_="foreignkey")
        batch_op.drop_constraint("fk_submissions_input_file_artifact", type_="foreignkey")
        batch_op.drop_index("ix_submissions_ingest_status")
        batch_op.drop_index("ix_submissions_uuid")
        batch_op.drop_column("ingest_status")
        batch_op.drop_column("failed_count")
        batch_op.drop_column("benchmark_yield_count")
        batch_op.drop_column("submitted_yield_count")
        batch_op.drop_column("row_count")
        batch_op.drop_column("input_filename")
        batch_op.drop_column("bovi_yields_artifact_id")
        batch_op.drop_column("submitted_yields_artifact_id")
        batch_op.drop_column("input_file_artifact_id")
        batch_op.drop_column("uuid")

    with op.batch_alter_table("challenges") as batch_op:
        batch_op.drop_constraint("fk_challenges_actual_yields_artifact", type_="foreignkey")
        batch_op.drop_constraint("fk_challenges_cow_metadata_artifact", type_="foreignkey")
        batch_op.drop_constraint("fk_challenges_actual_yields_file_artifact", type_="foreignkey")
        batch_op.drop_constraint("fk_challenges_test_day_file_artifact", type_="foreignkey")
        batch_op.drop_index("ix_challenges_ingest_status")
        batch_op.drop_index("ix_challenges_uuid")
        batch_op.drop_column("ingest_status")
        batch_op.drop_column("ingest_warnings")
        batch_op.drop_column("column_mapping")
        batch_op.drop_column("parity_counts")
        batch_op.drop_column("herd_count")
        batch_op.drop_column("actual_yield_count")
        batch_op.drop_column("cow_count")
        batch_op.drop_column("row_count")
        batch_op.drop_column("actual_yields_filename")
        batch_op.drop_column("test_day_filename")
        batch_op.drop_column("actual_yields_artifact_id")
        batch_op.drop_column("cow_metadata_artifact_id")
        batch_op.drop_column("actual_yields_file_artifact_id")
        batch_op.drop_column("test_day_file_artifact_id")
        batch_op.drop_column("uuid")

    op.drop_index("ix_uploaded_datasets_user_id", table_name="uploaded_datasets")
    op.drop_index("ix_uploaded_datasets_dataset_type", table_name="uploaded_datasets")
    op.drop_table("uploaded_datasets")

    op.drop_index("ix_storage_artifacts_blob_path", table_name="storage_artifacts")
    op.drop_index("ix_storage_artifacts_entity_uuid", table_name="storage_artifacts")
    op.drop_index("ix_storage_artifacts_entity_type", table_name="storage_artifacts")
    op.drop_index("ix_storage_artifacts_artifact_kind", table_name="storage_artifacts")
    op.drop_table("storage_artifacts")
