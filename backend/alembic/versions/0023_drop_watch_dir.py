"""drop dead watch_dir column from organisations

Revision ID: 0023_drop_watch_dir
Revises: 0022_event_counters
Create Date: 2026-08-14 00:00:00.000000

The watch_dir column was added in migration 0001 but is never referenced
in application code. It is dead and should be dropped.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0023_drop_watch_dir"
down_revision: str = "0022_event_counters"
branch_labels: str | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("organisations", "watch_dir")


def downgrade() -> None:
    op.add_column("organisations", sa.Column("watch_dir", sa.String(255), nullable=True))
