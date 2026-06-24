"""add SMB fields to organisations — DELETED: SMB share ingestion is not in the project roadmap.
Data ingestion is handled via mounted folders (/app/data/) monitored by the DataStore watcher.

This migration file is kept but does nothing to preserve the migration chain.
"""
from typing import Sequence, Union

from alembic import op

revision: str = '0002_add_smb_fields'
down_revision: Union[str, None] = '0001_add_watch_dir_to_organisations'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # No-op: SMB fields were never added to the database.
    pass


def downgrade() -> None:
    # No-op.
    pass
