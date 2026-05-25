"""add folders table, folder_id FK on chats, FULLTEXT index on messages.content

Revision ID: add_folders_and_message_search
Revises: add_branching_to_messages
Create Date: 2025-01-01

Adds:
  - folders table (id, name, user_id FK, created_at, updated_at)
  - chats.folder_id nullable FK -> folders.id ON DELETE SET NULL
  - FULLTEXT INDEX idx_messages_content ON messages(content)
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_folders_and_message_search'
down_revision = 'add_branching_to_messages'
branch_labels = None
depends_on = None


def _table_exists(conn, table_name):
    result = conn.execute(
        sa.text("SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = DATABASE() AND table_name = :t"),
        {"t": table_name},
    )
    return result.scalar() > 0


def _column_exists(conn, table_name, column_name):
    result = conn.execute(
        sa.text("SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_schema = DATABASE() AND table_name = :t AND column_name = :c"),
        {"t": table_name, "c": column_name},
    )
    return result.scalar() > 0


def _index_exists(conn, table_name, index_name):
    result = conn.execute(
        sa.text("SELECT COUNT(*) FROM information_schema.statistics "
                "WHERE table_schema = DATABASE() AND table_name = :t AND index_name = :i"),
        {"t": table_name, "i": index_name},
    )
    return result.scalar() > 0


def upgrade():
    conn = op.get_bind()

    # 1. Create folders table (idempotent)
    if not _table_exists(conn, 'folders'):
        op.create_table(
            'folders',
            sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
            sa.Column('name', sa.String(255), nullable=False),
            sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.Column('updated_at', sa.DateTime(), nullable=True),
        )
    if not _index_exists(conn, 'folders', 'ix_folders_user_id'):
        op.create_index('ix_folders_user_id', 'folders', ['user_id'])

    # 2. Add folder_id FK column to chats (idempotent)
    if not _column_exists(conn, 'chats', 'folder_id'):
        op.add_column(
            'chats',
            sa.Column('folder_id', sa.Integer(), sa.ForeignKey('folders.id', ondelete='SET NULL'), nullable=True),
        )
    if not _index_exists(conn, 'chats', 'ix_chats_folder_id'):
        op.create_index('ix_chats_folder_id', 'chats', ['folder_id'])

    # 3. Add FULLTEXT index on messages.content (idempotent)
    if not _index_exists(conn, 'messages', 'idx_messages_content'):
        op.execute(
            'ALTER TABLE messages ADD FULLTEXT INDEX idx_messages_content (content)'
        )


def downgrade():
    op.execute('ALTER TABLE messages DROP INDEX idx_messages_content')
    op.drop_index('ix_chats_folder_id', table_name='chats')
    op.drop_column('chats', 'folder_id')
    op.drop_index('ix_folders_user_id', table_name='folders')
    op.drop_table('folders')
