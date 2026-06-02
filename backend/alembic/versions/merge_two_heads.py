"""merge two heads

Revision ID: merge_two_heads
Revises: add_rewritten_query_to_messages, fd73eebc87c1
Create Date: 2026-06-02

"""
from alembic import op
import sqlalchemy as sa

revision = "merge_two_heads"
down_revision = ("add_rewritten_query_to_messages", "fd73eebc87c1")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
