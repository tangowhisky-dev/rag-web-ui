"""add enterprise agent support

Adds structured JSON columns to ``messages`` for the agent loop,
and creates the ``tool_call_audit`` table for enterprise observability.

Revision ID: 0017_enterprise_agent
Revises: 0016_add_document_file_path_datastore_uq
Create Date: 2026-07-27

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0017_enterprise_agent"
down_revision: Union[str, None] = "0016_add_document_file_path_datastore_uq"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("last_answer_object", sa.JSON(), nullable=True))
    op.add_column("messages", sa.Column("plan", sa.JSON(), nullable=True))
    op.add_column("messages", sa.Column("tool_calls", sa.JSON(), nullable=True))

    op.create_table(
        "tool_call_audit",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "chat_id",
            sa.Integer(),
            sa.ForeignKey("chats.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "message_id",
            sa.Integer(),
            sa.ForeignKey("messages.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column("iteration", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tool_name", sa.String(255), nullable=False),
        sa.Column("arguments", sa.JSON(), nullable=True),
        sa.Column("result_summary", sa.JSON(), nullable=True),
        sa.Column("tokens_in", sa.Integer(), nullable=True),
        sa.Column("tokens_out", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(
        "ix_tool_call_audit_chat_created",
        "tool_call_audit",
        ["chat_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_tool_call_audit_chat_created", table_name="tool_call_audit")
    op.drop_table("tool_call_audit")
    op.drop_column("messages", "tool_calls")
    op.drop_column("messages", "plan")
    op.drop_column("messages", "last_answer_object")
