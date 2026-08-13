"""add event-driven processing counters to datastores

Revision ID: 0022_event_counters
Revises: 0021_unique_doc_uploads
Create Date: 2026-08-14 00:00:00.000000

Splits last_scan_processed (previously overloaded by both manual scans and
event-driven processing) into a separate event-driven counter so the UI
can distinguish the two.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0022_event_counters"
down_revision: str = "0021_unique_doc_uploads"
branch_labels: str | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("data_stores", sa.Column("last_event_processed", sa.Integer(), nullable=True, server_default="0"))
    op.add_column("data_stores", sa.Column("last_event_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("data_stores", "last_event_at")
    op.drop_column("data_stores", "last_event_processed")
