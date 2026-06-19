"""add scan result columns to data_stores

Adds last_scan_new, last_scan_modified, last_scan_skipped, and
last_scan_errors columns to the data_stores table so that scan results
are persisted and available after the in-memory scan entry is evicted.

Revision ID: add_scan_result_cols
Revises: f605e1572096
Create date: 2026-06-18
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "add_scan_result_cols"
down_revision: Union[str, None] = "f605e1572096"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("data_stores", sa.Column("last_scan_new", sa.Integer(), nullable=True, server_default="0"))
    op.add_column("data_stores", sa.Column("last_scan_modified", sa.Integer(), nullable=True, server_default="0"))
    op.add_column("data_stores", sa.Column("last_scan_skipped", sa.Integer(), nullable=True, server_default="0"))
    op.add_column("data_stores", sa.Column("last_scan_errors", sa.Integer(), nullable=True, server_default="0"))


def downgrade() -> None:
    op.drop_column("data_stores", "last_scan_errors")
    op.drop_column("data_stores", "last_scan_skipped")
    op.drop_column("data_stores", "last_scan_modified")
    op.drop_column("data_stores", "last_scan_new")