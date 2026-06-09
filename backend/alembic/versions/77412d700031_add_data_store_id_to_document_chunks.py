"""add data_store_id to document_chunks

Revision ID: 77412d700031
Revises: 0010_fix_doc_datastore_cascade
Create Date: 2026-06-09

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '77412d700031'
down_revision = '0010_fix_doc_datastore_cascade'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add data_store_id column to document_chunks (nullable, no index yet)
    op.add_column('document_chunks', sa.Column('data_store_id', sa.Integer(), nullable=True))
    
    # Make kb_id nullable — DataStore documents have kb_id=Null
    op.alter_column('document_chunks', 'kb_id', existing_type=sa.Integer(), nullable=True)
    
    # Create index on data_store_id for DataStore chunk lookups
    op.create_index('ix_data_store_id', 'document_chunks', ['data_store_id'])


def downgrade() -> None:
    op.drop_index('ix_data_store_id', table_name='document_chunks')
    op.alter_column('document_chunks', 'kb_id', existing_type=sa.Integer(), nullable=False)
    op.drop_column('document_chunks', 'data_store_id')
