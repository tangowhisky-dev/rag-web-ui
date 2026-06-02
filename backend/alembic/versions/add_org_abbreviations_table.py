"""add org abbreviations table

Revision ID: add_org_abbreviations_table
Revises: add_org_llm_config
Create Date: 2026-06-02

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "add_org_abbreviations_table"
down_revision: Union[str, None] = "add_org_llm_config"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "org_abbreviations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("org_id", sa.Integer(), nullable=False),
        sa.Column("short", sa.String(64), nullable=False),
        sa.Column("expansion", sa.String(512), nullable=False),
        sa.ForeignKeyConstraint(["org_id"], ["organisations.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("org_id", "short", name="uq_org_abbreviations_org_short"),
    )
    op.create_index("ix_org_abbreviations_org_id", "org_abbreviations", ["org_id"])


def downgrade() -> None:
    op.drop_index("ix_org_abbreviations_org_id", table_name="org_abbreviations")
    op.drop_table("org_abbreviations")
