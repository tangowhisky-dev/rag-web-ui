"""add org llm config

Revision ID: add_org_llm_config
Revises: add_organisations_and_roles
Create Date: 2026-06-02

"""
from alembic import op
import sqlalchemy as sa

revision = "add_org_llm_config"
down_revision = "add_org_abbreviations_table"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "org_llm_configs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "org_id",
            sa.Integer(),
            sa.ForeignKey("organisations.id"),
            nullable=False,
        ),
        sa.Column("api_base", sa.String(512), nullable=True),
        sa.Column("model_name", sa.String(255), nullable=True),
        sa.Column("query_model", sa.String(255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_org_llm_configs_id", "org_llm_configs", ["id"])
    op.create_index("ix_org_llm_configs_org_id", "org_llm_configs", ["org_id"])
    op.create_unique_constraint("uq_org_llm_configs_org_id", "org_llm_configs", ["org_id"])


def downgrade() -> None:
    op.drop_table("org_llm_configs")
