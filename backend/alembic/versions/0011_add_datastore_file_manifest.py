"""add datastore file manifest table

Creates the data_store_file_manifests table to track every file known
to a datastore with its SHA-256 hash and file size.

Revision ID: 0011_add_datastore_file_manifest
Revises: add_scan_result_cols
Create Date: 2026-06-23

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0011_add_datastore_file_manifest"
down_revision: Union[str, None] = "add_scan_result_cols"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "data_store_file_manifests",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("datastore_id", sa.Integer(), nullable=False, index=True),
        sa.Column("file_path", sa.String(length=512), nullable=False),
        sa.Column("file_hash", sa.String(length=64), nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("discovered_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["datastore_id"],
            ["data_stores.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "datastore_id", "file_path", name="uq_datastore_file_path"
        ),
    )


def downgrade() -> None:
    op.drop_table("data_store_file_manifests")
