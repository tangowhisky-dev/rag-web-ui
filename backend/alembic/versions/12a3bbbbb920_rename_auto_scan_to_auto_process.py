"""rename auto_scan to auto_process

Revision ID: 12a3bbbbb920
Revises: 70b6bf55716c
Create Date: 2026-08-25 10:09:36.692084

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '12a3bbbbb920'
down_revision: Union[str, None] = '70b6bf55716c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column('data_stores', 'auto_scan_enabled',
                    new_column_name='auto_process_enabled',
                    existing_type=sa.Boolean(),
                    existing_nullable=True)
    op.alter_column('data_stores', 'auto_scan_interval_minutes',
                    new_column_name='auto_process_interval_minutes',
                    existing_type=sa.Integer(),
                    existing_nullable=True)


def downgrade() -> None:
    op.alter_column('data_stores', 'auto_process_enabled',
                    new_column_name='auto_scan_enabled',
                    existing_type=sa.Boolean(),
                    existing_nullable=True)
    op.alter_column('data_stores', 'auto_process_interval_minutes',
                    new_column_name='auto_scan_interval_minutes',
                    existing_type=sa.Integer(),
                    existing_nullable=True)
