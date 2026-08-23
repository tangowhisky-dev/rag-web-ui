"""Add modified_at column to documents for recency-aware dedup.

Revision ID: 0029_add_document_modified_at
Revises: 0028_add_cascade_to_task_chunk_datastore_fk
Create Date: 2026-08-22

Stores the source file's modification timestamp at ingestion time.
Used by retrieval dedup to prefer chunks from the latest document version
when duplicate or near-duplicate content is found across documents.

Nullable — backfilled to created_at for existing rows so COALESCE always
returns a value.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0029_add_document_modified_at'
down_revision = '0028_add_cascade_to_task_chunk_datastore_fk'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'documents',
        sa.Column('modified_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_documents_modified_at', 'documents', ['modified_at'])
    op.execute(
        "UPDATE documents SET modified_at = created_at WHERE modified_at IS NULL"
    )


def downgrade() -> None:
    op.drop_index('ix_documents_modified_at', table_name='documents')
    op.drop_column('documents', 'modified_at')
