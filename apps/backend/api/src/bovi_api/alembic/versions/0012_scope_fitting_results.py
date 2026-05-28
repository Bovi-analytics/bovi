"""scope fitting results to users and organizations

Revision ID: 0012
Revises: 0011
Create Date: 2026-05-28
"""

import sqlalchemy as sa
from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("fitting_results") as batch_op:
        batch_op.add_column(sa.Column("user_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("organization_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_fitting_results_user_id_users",
            "users",
            ["user_id"],
            ["id"],
        )
        batch_op.create_foreign_key(
            "fk_fitting_results_organization_id_organizations",
            "organizations",
            ["organization_id"],
            ["id"],
        )
    op.create_index(op.f("ix_fitting_results_user_id"), "fitting_results", ["user_id"])
    op.create_index(
        op.f("ix_fitting_results_organization_id"),
        "fitting_results",
        ["organization_id"],
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_fitting_results_organization_id"), table_name="fitting_results")
    op.drop_index(op.f("ix_fitting_results_user_id"), table_name="fitting_results")
    with op.batch_alter_table("fitting_results") as batch_op:
        batch_op.drop_constraint(
            "fk_fitting_results_organization_id_organizations", type_="foreignkey"
        )
        batch_op.drop_constraint("fk_fitting_results_user_id_users", type_="foreignkey")
        batch_op.drop_column("organization_id")
        batch_op.drop_column("user_id")
