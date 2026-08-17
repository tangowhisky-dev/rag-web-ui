"""messages parent_message_id FK ondelete CASCADE

Revision ID: 0025_messages_parent_fk_cascade
Revises: 0024_add_active_branches_to_chats
Create Date: 2026-08-17

Changes the self-referencing FK on messages.parent_message_id from
the default RESTRICT to CASCADE. Without this, deleting a chat fails
when messages reference each other via parent_message_id — MySQL
refuses to delete a parent row while child rows still reference it.
"""
from alembic import op

# revision identifiers, used by Alembic.
revision = '0025_messages_parent_fk_cascade'
down_revision = '0024_add_active_branches_to_chats'
branch_labels = None
depends_on = None


def upgrade():
    op.drop_constraint(
        'messages_ibfk_2',
        'messages',
        type_='foreignkey',
    )
    op.create_foreign_key(
        'messages_ibfk_2',
        'messages',
        'messages',
        ['parent_message_id'],
        ['id'],
        ondelete='CASCADE',
    )


def downgrade():
    op.drop_constraint('messages_ibfk_2', 'messages', type_='foreignkey')
    op.create_foreign_key(
        'messages_ibfk_2',
        'messages',
        'messages',
        ['parent_message_id'],
        ['id'],
    )
