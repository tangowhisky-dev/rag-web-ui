"""add watch_dir to organisations

Revision ID: 0001_add_watch_dir
Revises: merge_two_heads
Create Date: 2026-06-03 05:07:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '0001_add_watch_dir'
down_revision: Union[str, None] = 'merge_two_heads'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'organisations',
        sa.Column('watch_dir', sa.String(length=1024), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('organisations', 'watch_dir')
