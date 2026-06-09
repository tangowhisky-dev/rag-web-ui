"""add data_store_id to processing_tasks

Revision ID: f605e1572096
Revises: 77412d700031
Create Date: 2026-06-09

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f605e1572096'
down_revision = '77412d700031'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Make knowledge_base_id nullable — DataStore tasks have knowledge_base_id=NULL
    op.alter_column('processing_tasks', 'knowledge_base_id', existing_type=sa.Integer(), nullable=True)
    
    # Add data_store_id column to processing_tasks (nullable)
    op.add_column('processing_tasks', sa.Column('data_store_id', sa.Integer(), nullable=True))
    
    # Create index on data_store_id for DataStore task lookups
    op.create_index('ix_data_store_id', 'processing_tasks', ['data_store_id'])


def downgrade() -> None:
    op.drop_index('ix_data_store_id', table_name='processing_tasks')
    op.drop_column('processing_tasks', 'data_store_id')
    op.alter_column('processing_tasks', 'knowledge_base_id', existing_type=sa.Integer(), nullable=False)
