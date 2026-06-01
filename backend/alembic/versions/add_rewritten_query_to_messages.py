"""add rewritten_query to messages

Revision ID: add_rewritten_query_to_messages
Revises: add_retrieval_leg_flags_to_chats
Create Date: 2026-06-01
"""
from alembic import op
import sqlalchemy as sa

revision = "add_rewritten_query_to_messages"
down_revision = "add_folders_and_message_search"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "messages",
        sa.Column("rewritten_query", sa.Text(), nullable=True),
    )


def downgrade():
    op.drop_column("messages", "rewritten_query")
