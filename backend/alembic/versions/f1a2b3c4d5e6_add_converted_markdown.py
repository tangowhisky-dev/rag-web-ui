"""add converted_markdown, conversion_status, conversion_error, lock_version

Revision ID: f1a2b3c4d5e6
Revises: e6f7a8b9c0d1
Create Date: 2026-02-27 12:00:00.000000

Three-phase ingestion pipeline: persist the converted markdown so admins
can review and correct OCR/conversion artifacts before re-ingesting.

Columns added to ``documents``:
- ``converted_markdown`` LONGTEXT — the output of phase 1 (file → markdown)
- ``conversion_status`` VARCHAR(20) — pending / completed / error / NULL(legacy)
- ``conversion_error`` TEXT — failure message when conversion_status='error'
- ``lock_version`` INTEGER NOT NULL DEFAULT 0 — optimistic locking for editor

Legacy documents get conversion_status=NULL; the recovery service queues
conversion for them on startup.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import LONGTEXT


revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, None] = 'e6f7a8b9c0d1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'documents',
        sa.Column('converted_markdown', LONGTEXT(), nullable=True),
    )
    op.add_column(
        'documents',
        sa.Column('conversion_status', sa.String(20), nullable=True, index=True),
    )
    op.add_column(
        'documents',
        sa.Column('conversion_error', sa.Text(), nullable=True),
    )
    op.add_column(
        'documents',
        sa.Column('lock_version', sa.Integer(), nullable=False, server_default='0'),
    )


def downgrade() -> None:
    op.drop_column('documents', 'lock_version')
    op.drop_column('documents', 'conversion_error')
    op.drop_index('ix_documents_conversion_status', table_name='documents')
    op.drop_column('documents', 'conversion_status')
    op.drop_column('documents', 'converted_markdown')
