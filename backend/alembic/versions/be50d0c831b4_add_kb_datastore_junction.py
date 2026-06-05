"""add_kb_datastore_junction

Revision ID: be50d0c831b4
Revises: c0a76636ff83
Create Date: 2026-06-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'be50d0c831b4'
down_revision: Union[str, None] = 'c0a76636ff83'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create junction table for KB-DataStore relationships
    op.create_table(
        'knowledge_base_datastores',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('knowledge_base_id', sa.Integer(), nullable=False),
        sa.Column('data_store_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['knowledge_base_id'], ['knowledge_bases.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['data_store_id'], ['data_stores.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('knowledge_base_id', 'data_store_id'),
    )
    op.create_index('ix_kb_datastores_kb_id', 'knowledge_base_datastores', ['knowledge_base_id'])
    op.create_index('ix_kb_datastores_ds_id', 'knowledge_base_datastores', ['data_store_id'])


def downgrade() -> None:
    op.drop_index('ix_kb_datastores_ds_id')
    op.drop_index('ix_kb_datastores_kb_id')
    op.drop_table('knowledge_base_datastores')
