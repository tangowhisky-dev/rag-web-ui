"""add active_branches to chats

Revision ID: 0024_add_active_branches_to_chats
Revises: 0023_drop_watch_dir
Create Date: 2026-08-13

Adds `active_branches` (JSON, nullable) to the `chats` table to support
conversation branching. The model already declares this column but no
prior migration created it.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import JSON


# revision identifiers, used by Alembic.
revision = '0024_add_active_branches_to_chats'
down_revision = '0023_drop_watch_dir'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'chats',
        sa.Column('active_branches', JSON(), nullable=True),
    )


def downgrade():
    op.drop_column('chats', 'active_branches')
