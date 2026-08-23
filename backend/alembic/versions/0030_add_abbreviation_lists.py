"""Add abbreviation_lists and abbreviations tables; drop org_abbreviations.

Revision ID: 0030_add_abbreviation_lists
Revises: 0029_add_document_modified_at
Create Date: 2026-09-24
"""
from alembic import op
import sqlalchemy as sa

revision = "0030_add_abbreviation_lists"
down_revision = "0029_add_document_modified_at"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop the old dead-code table
    op.drop_index("ix_org_abbreviations_org_id", table_name="org_abbreviations")
    op.drop_table("org_abbreviations")

    op.create_table(
        "abbreviation_lists",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("org_id", sa.Integer(), nullable=True),
        sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("row_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_by", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["org_id"], ["organisations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_abbrev_lists_org", "abbreviation_lists", ["org_id"])
    op.create_index("idx_abbrev_lists_enabled", "abbreviation_lists", ["is_enabled"])

    op.create_table(
        "abbreviations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("list_id", sa.Integer(), nullable=False),
        sa.Column("abbreviation", sa.String(64), nullable=False),
        sa.Column("expanded_form", sa.String(512), nullable=False),
        sa.Column("category", sa.String(255), nullable=True),
        sa.ForeignKeyConstraint(["list_id"], ["abbreviation_lists.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_abbreviations_list", "abbreviations", ["list_id"])
    op.create_index("idx_abbreviations_abbr", "abbreviations", ["abbreviation"])


def downgrade() -> None:
    op.drop_index("idx_abbreviations_abbr", table_name="abbreviations")
    op.drop_index("idx_abbreviations_list", table_name="abbreviations")
    op.drop_table("abbreviations")
    op.drop_index("idx_abbrev_lists_enabled", table_name="abbreviation_lists")
    op.drop_index("idx_abbrev_lists_org", table_name="abbreviation_lists")
    op.drop_table("abbreviation_lists")

    # Recreate old table
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
