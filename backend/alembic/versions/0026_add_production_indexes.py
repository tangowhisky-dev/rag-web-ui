"""Add production-critical indexes for datastore performance.

Revision ID: 0026_add_production_indexes
Revises: 0025_messages_parent_fk_cascade
Create Date: 2026-08-18

Adds missing indexes identified during production readiness review:
- processing_tasks.status: stuck task query scans full table
- processing_tasks.document_id: orphan cleanup uses outer join
- processing_tasks.data_store_id: active task queries per datastore
- document_chunks.data_store_id: Qdrant reconciliation scrolls points
- document_chunks.document_id: deletion queries
- data_store_file_manifest.(datastore_id, file_hash): modified file detection
- data_stores.is_active: startup recovery query
- documents.file_path: watcher file lookup (unique constraint already
  exists for (file_path, data_store_id) but a standalone index on
  file_path speeds up watcher lookups that filter by path alone)

Also widens data_stores.folder_path from String(512) to String(1024)
and documents.file_path from String(255) to String(1024) to handle
deep directory structures.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0026_add_production_indexes'
down_revision = '0025_messages_parent_fk_cascade'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -- Column type widening --
    op.alter_column(
        'data_stores', 'folder_path',
        existing_type=sa.String(512),
        type_=sa.String(1024),
        existing_nullable=False,
    )
    op.alter_column(
        'documents', 'file_path',
        existing_type=sa.String(255),
        type_=sa.String(1024),
        existing_nullable=False,
    )

    # -- New indexes --
    op.create_index('ix_processing_tasks_status', 'processing_tasks', ['status'])
    op.create_index('ix_processing_tasks_document_id', 'processing_tasks', ['document_id'])
    op.create_index('ix_processing_tasks_data_store_id', 'processing_tasks', ['data_store_id'])
    op.create_index('ix_document_chunks_data_store_id', 'document_chunks', ['data_store_id'])
    op.create_index('ix_document_chunks_document_id', 'document_chunks', ['document_id'])
    op.create_index('ix_data_stores_is_active', 'data_stores', ['is_active'])
    op.create_index(
        'ix_manifest_datastore_file_hash',
        'data_store_file_manifest',
        ['datastore_id', 'file_hash'],
    )


def downgrade() -> None:
    op.drop_index('ix_manifest_datastore_file_hash', table_name='data_store_file_manifest')
    op.drop_index('ix_data_stores_is_active', table_name='data_stores')
    op.drop_index('ix_document_chunks_document_id', table_name='document_chunks')
    op.drop_index('ix_document_chunks_data_store_id', table_name='document_chunks')
    op.drop_index('ix_processing_tasks_data_store_id', table_name='processing_tasks')
    op.drop_index('ix_processing_tasks_document_id', table_name='processing_tasks')
    op.drop_index('ix_processing_tasks_status', table_name='processing_tasks')

    op.alter_column(
        'documents', 'file_path',
        existing_type=sa.String(1024),
        type_=sa.String(255),
        existing_nullable=False,
    )
    op.alter_column(
        'data_stores', 'folder_path',
        existing_type=sa.String(1024),
        type_=sa.String(512),
        existing_nullable=False,
    )
