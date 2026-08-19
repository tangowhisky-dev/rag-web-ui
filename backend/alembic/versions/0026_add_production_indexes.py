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
    conn = op.get_bind()

    # -- Column type widening --
    # data_stores.folder_path has a UNIQUE constraint.  With utf8mb4 encoding,
    # VARCHAR(1024) = 4096 bytes, exceeding MySQL's 3072-byte index key limit.
    # Drop the existing unique index, widen the column, then recreate the
    # unique index with a 768-char prefix (768 × 4 = 3072 bytes, the max).
    # Use raw SQL for index inspection/dropping because the index name varies
    # (SQLAlchemy auto-generates names like ix_data_stores_folder_path).
    result = conn.execute(sa.text(
        "SELECT INDEX_NAME FROM information_schema.STATISTICS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'data_stores' "
        "AND COLUMN_NAME = 'folder_path'"
    ))
    for row in result:
        idx_name = row[0]
        conn.execute(sa.text(f"DROP INDEX `{idx_name}` ON data_stores"))

    op.alter_column(
        'data_stores', 'folder_path',
        existing_type=sa.String(512),
        type_=sa.String(1024),
        existing_nullable=False,
    )
    op.create_index(
        'ix_data_stores_folder_path',
        'data_stores',
        ['folder_path'],
        unique=True,
        mysql_length=767,
    )

    # documents.file_path has a composite unique constraint
    # (file_path, data_store_id).  With utf8mb4, VARCHAR(1024) = 4096 bytes,
    # plus the INT data_store_id = 4100 bytes, exceeding the 3072-byte limit.
    # Drop the constraint, widen the column, recreate with a 768-char prefix
    # on file_path.
    result = conn.execute(sa.text(
        "SELECT INDEX_NAME FROM information_schema.STATISTICS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'documents' "
        "AND COLUMN_NAME = 'file_path'"
    ))
    for row in result:
        idx_name = row[0]
        conn.execute(sa.text(f"DROP INDEX `{idx_name}` ON documents"))

    op.alter_column(
        'documents', 'file_path',
        existing_type=sa.String(255),
        type_=sa.String(1024),
        existing_nullable=False,
    )
    # Recreate composite unique constraint with prefix on file_path.
    # mysql_length accepts a dict mapping column names to prefix lengths.
    op.create_index(
        'uq_document_file_path_datastore',
        'documents',
        ['file_path', 'data_store_id'],
        unique=True,
        mysql_length={'file_path': 767},
    )

    # -- New indexes (idempotent — skip if already exists) --
    def _index_exists(table: str, index_name: str) -> bool:
        result = conn.execute(sa.text(
            "SELECT COUNT(*) FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :table "
            "AND INDEX_NAME = :index_name"
        ), {"table": table, "index_name": index_name})
        return result.scalar() > 0

    indexes_to_create = [
        ('ix_processing_tasks_status', 'processing_tasks', ['status']),
        ('ix_processing_tasks_document_id', 'processing_tasks', ['document_id']),
        ('ix_processing_tasks_data_store_id', 'processing_tasks', ['data_store_id']),
        ('ix_document_chunks_data_store_id', 'document_chunks', ['data_store_id']),
        ('ix_document_chunks_document_id', 'document_chunks', ['document_id']),
        ('ix_data_stores_is_active', 'data_stores', ['is_active']),
        ('ix_manifest_datastore_file_hash', 'data_store_file_manifests', ['datastore_id', 'file_hash']),
    ]
    for idx_name, table, cols in indexes_to_create:
        if not _index_exists(table, idx_name):
            op.create_index(idx_name, table, cols)


def downgrade() -> None:
    op.drop_index('ix_manifest_datastore_file_hash', table_name='data_store_file_manifests')
    op.drop_index('ix_data_stores_is_active', table_name='data_stores')
    op.drop_index('ix_document_chunks_document_id', table_name='document_chunks')
    op.drop_index('ix_document_chunks_data_store_id', table_name='document_chunks')
    op.drop_index('ix_processing_tasks_data_store_id', table_name='processing_tasks')
    op.drop_index('ix_processing_tasks_document_id', table_name='processing_tasks')
    op.drop_index('ix_processing_tasks_status', table_name='processing_tasks')

    # Restore documents.file_path to 255 and recreate original unique constraint
    op.drop_index('uq_document_file_path_datastore', table_name='documents')
    op.alter_column(
        'documents', 'file_path',
        existing_type=sa.String(1024),
        type_=sa.String(255),
        existing_nullable=False,
    )
    op.create_unique_constraint(
        'uq_document_file_path_datastore', 'documents',
        ['file_path', 'data_store_id'],
    )

    # Restore data_stores.folder_path to 512 and recreate original unique constraint
    op.drop_index('ix_data_stores_folder_path', table_name='data_stores')
    op.alter_column(
        'data_stores', 'folder_path',
        existing_type=sa.String(1024),
        type_=sa.String(512),
        existing_nullable=False,
    )
    op.create_unique_constraint('folder_path', 'data_stores', ['folder_path'])
