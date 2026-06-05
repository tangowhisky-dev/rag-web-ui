"""add datastores

Revision ID: 0008_add_datastores
Revises: merge_two_heads
Create Date: 2026-06-04

Adds DataStore and OrganizationDataStore tables to replace the
per-organisation watch_dir with a shared, first-class datastore model.
"""

from alembic import op
import sqlalchemy as sa

revision = "0008_add_datastores"
down_revision = "0007_fix_fk_cascade"
branch_labels = None
depends_on = None


def upgrade():
    # --- data_stores table ---
    op.create_table(
        "data_stores",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("folder_path", sa.String(512), nullable=False, unique=True),
        sa.Column("scan_pattern", sa.String(100), server_default="*"),
        sa.Column("is_active", sa.Boolean(), server_default="1"),
        sa.Column("auto_scan_enabled", sa.Boolean(), server_default="0"),
        sa.Column("auto_scan_interval_minutes", sa.Integer(), server_default="60"),
        sa.Column("last_scan_at", sa.DateTime(), nullable=True),
        sa.Column("last_scan_status", sa.String(50), server_default="never"),
        sa.Column("last_scan_error", sa.Text(), nullable=True),
        sa.Column("last_scan_total_files", sa.Integer(), server_default="0"),
        sa.Column("last_scan_processed", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )

    # --- organization_data_stores junction table ---
    op.create_table(
        "organization_data_stores",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("org_id", sa.Integer(), sa.ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("data_store_id", sa.Integer(), sa.ForeignKey("data_stores.id", ondelete="CASCADE"), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default="1"),
        sa.Column("assigned_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.UniqueConstraint("org_id", "data_store_id", name="uq_org_datastore"),
    )

    # --- Add data_store_id to documents table ---
    op.add_column(
        "documents",
        sa.Column("data_store_id", sa.Integer(), nullable=True, index=True),
    )
    op.create_foreign_key(
        "fk_documents_data_store_id",
        "documents",
        "data_stores",
        ["data_store_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade():
    op.drop_constraint("fk_documents_data_store_id", "documents", type_="foreignkey")
    op.drop_column("documents", "data_store_id")
    op.drop_table("organization_data_stores")
    op.drop_table("data_stores")
