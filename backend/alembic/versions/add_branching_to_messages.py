"""add parent_message_id and branch_index to messages

Revision ID: add_branching_to_messages
Revises: a1b2c3d4e5f7, e3f4a5b6c7d8
Create Date: 2025-01-01

Adds `parent_message_id` (nullable FK to messages.id) and `branch_index`
(integer NOT NULL DEFAULT 0) to the `messages` table to support conversation
branching when a user edits a sent message.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_branching_to_messages'
down_revision = ('a1b2c3d4e5f7', 'e3f4a5b6c7d8')
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'messages',
        sa.Column('parent_message_id', sa.Integer(), sa.ForeignKey('messages.id'), nullable=True),
    )
    op.add_column(
        'messages',
        sa.Column('branch_index', sa.Integer(), nullable=False, server_default='0'),
    )
    op.create_index(
        'ix_messages_parent_message_id',
        'messages',
        ['parent_message_id'],
    )


def downgrade():
    op.drop_index('ix_messages_parent_message_id', table_name='messages')
    op.drop_column('messages', 'branch_index')
    op.drop_column('messages', 'parent_message_id')
