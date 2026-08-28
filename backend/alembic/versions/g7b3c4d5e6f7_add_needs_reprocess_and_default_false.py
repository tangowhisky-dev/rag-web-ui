"""add needs_reprocess column, change is_selected default to FALSE

Revision ID: g7b3c4d5e6f7
Revises: f1a2b3c4d5e6
Create Date: 2026-08-28 12:00:00.000000

Two changes:
1. Add documents.needs_reprocess (Boolean, NOT NULL, default FALSE).
   Set when an admin edits converted markdown; cleared after
   successful re-ingest.
2. Change is_selected server_default from '1' to '0'.  Existing
   rows keep their current value — only NEW rows default to
   unselected.  This matches the new model where files must be
   explicitly selected before processing.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'g7b3c4d5e6f7'
down_revision: Union[str, None] = 'f1a2b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add needs_reprocess column
    op.add_column(
        'documents',
        sa.Column('needs_reprocess', sa.Boolean(), nullable=False, server_default=sa.text('0')),
    )

    # 2. Change is_selected default to FALSE for new rows.
    #    Existing rows keep their current value.
    op.alter_column(
        'documents', 'is_selected',
        existing_type=sa.Boolean(),
        server_default=sa.text('0'),
    )


def downgrade() -> None:
    op.alter_column(
        'documents', 'is_selected',
        existing_type=sa.Boolean(),
        server_default=sa.text('1'),
    )
    op.drop_column('documents', 'needs_reprocess')
