"""add graph_ingestion_paused column

Revision ID: a8b2c3d4e5f6
Revises: 12a3bbbbb920
Create Date: 2026-08-25 12:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a8b2c3d4e5f6'
down_revision: Union[str, None] = '12a3bbbbb920'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'data_stores',
        sa.Column('graph_ingestion_paused', sa.Boolean(), nullable=True, server_default=sa.text('0')),
    )


def downgrade() -> None:
    op.drop_column('data_stores', 'graph_ingestion_paused')
