"""Add ON DELETE CASCADE FK constraints for data_store_id columns.

Revision ID: 0028_add_cascade_to_task_chunk_datastore_fk
Revises: 0027_add_manifest_file_mtime
Create Date: 2026-08-20

ProcessingTask.data_store_id and DocumentChunk.data_store_id had no
database-level FK constraint (the ORM declared ForeignKey but earlier
migrations never created the constraints).  When a DataStore was deleted,
orphaned task and chunk rows with dangling data_store_id values remained.

This migration adds ON DELETE CASCADE foreign key constraints to both
columns so that deleting a DataStore automatically removes its tasks
and chunks at the database level.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0028_add_cascade_to_task_chunk_datastore_fk'
down_revision = '0027_add_manifest_file_mtime'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_foreign_key(
        'fk_processing_tasks_data_store_id',
        'processing_tasks',
        'data_stores',
        ['data_store_id'],
        ['id'],
        ondelete='CASCADE',
    )

    op.create_foreign_key(
        'fk_document_chunks_data_store_id',
        'document_chunks',
        'data_stores',
        ['data_store_id'],
        ['id'],
        ondelete='CASCADE',
    )


def downgrade() -> None:
    op.drop_constraint('fk_document_chunks_data_store_id', 'document_chunks', type_='foreignkey')
    op.drop_constraint('fk_processing_tasks_data_store_id', 'processing_tasks', type_='foreignkey')
