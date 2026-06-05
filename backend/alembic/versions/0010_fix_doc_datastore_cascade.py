"""fix document data_store_id FK to CASCADE

Revision ID: 0010_fix_doc_datastore_cascade
Revises: 0009_conditional_doc_delete
Create Date: 2026-06-05

Change documents.data_store_id FK from SET NULL to CASCADE.
When a DataStore is deleted, its documents should also be deleted from DB.
Actual files on disk are NOT deleted - they remain in the DataStore folder.

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0010_fix_doc_datastore_cascade'
down_revision = '0009_conditional_doc_delete'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop existing FK constraint (named fk_documents_data_store_id)
    op.drop_constraint('fk_documents_data_store_id', 'documents', type_='foreignkey')
    
    # Recreate with CASCADE (delete documents when DataStore is deleted)
    op.create_foreign_key(
        'fk_documents_data_store_id',
        'documents', 'data_stores',
        ['data_store_id'], ['id'],
        ondelete='CASCADE'
    )


def downgrade() -> None:
    # Drop the new constraint
    op.drop_constraint('fk_documents_data_store_id', 'documents', type_='foreignkey')
    
    # Recreate with SET NULL (original behavior after migration 0009)
    op.create_foreign_key(
        'fk_documents_data_store_id',
        'documents', 'data_stores',
        ['data_store_id'], ['id'],
        ondelete='SET NULL'
    )
