"""add documents.is_selected column

Revision ID: e6f7a8b9c0d1
Revises: d5e6f7a8b9c1
Create Date: 2026-08-27 10:00:00.000000

Adds a non-null boolean column to documents controlling whether a
file participates in ingestion/reingestion.  Defaults to TRUE so
existing documents remain active — no behaviour change for current
datastores.  A composite index on (data_store_id, is_selected)
supports efficient filtered queries from the browse API.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e6f7a8b9c0d1'
down_revision: Union[str, None] = 'd5e6f7a8b9c1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'documents',
        sa.Column('is_selected', sa.Boolean(), nullable=False, server_default=sa.text('1')),
    )
    op.create_index(
        'ix_documents_datastore_selected',
        'documents',
        ['data_store_id', 'is_selected'],
    )


def downgrade() -> None:
    op.drop_index('ix_documents_datastore_selected', table_name='documents')
    op.drop_column('documents', 'is_selected')
