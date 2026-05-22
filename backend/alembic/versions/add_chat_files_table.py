"""add chat_files table for chat-scoped file uploads

Revision ID: f1a2b3c4d5e6
Revises: c4d5e6f7a8b9
Create Date: 2025-05-22
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import LONGTEXT

revision = "f1a2b3c4d5e6"
down_revision = "c4d5e6f7a8b9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_files",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("chat_id", sa.Integer(), sa.ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("message_id", sa.Integer(), sa.ForeignKey("messages.id", ondelete="SET NULL"), nullable=True, index=True),
        sa.Column("file_name", sa.String(255), nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("content_type", sa.String(100), nullable=False),
        sa.Column("markdown_content", LONGTEXT(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="processing"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("now()"), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("chat_files")
