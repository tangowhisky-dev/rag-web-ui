"""drop chat retrieval leg flags

Revision ID: e4f5a6b7c8d9
Revises: 0014_add_clarification_requests_and_user_profiles
Create Date: 2026-07-13

Removes per-chat retrieval toggles from `chats`:
  use_graph_rag, use_dense, use_sparse, use_exact

All chats now use every globally enabled retrieval source.
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "e4f5a6b7c8d9"
down_revision: Union[str, None] = "0014_add_clarification_requests_and_user_profiles"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("chats", "use_exact")
    op.drop_column("chats", "use_sparse")
    op.drop_column("chats", "use_dense")
    op.drop_column("chats", "use_graph_rag")


def downgrade() -> None:
    op.add_column("chats", sa.Column("use_graph_rag", sa.Boolean(), nullable=False, server_default="0"))
    op.add_column("chats", sa.Column("use_dense", sa.Boolean(), nullable=False, server_default="1"))
    op.add_column("chats", sa.Column("use_sparse", sa.Boolean(), nullable=False, server_default="1"))
    op.add_column("chats", sa.Column("use_exact", sa.Boolean(), nullable=False, server_default="1"))
