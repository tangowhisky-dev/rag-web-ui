"""drop is_superuser column from users table

Revision ID: 0003_drop_is_superuser
Revises: merge_two_heads
Create Date: 2026-06-04

"""
from alembic import op
import sqlalchemy as sa

revision = "0003_drop_is_superuser"
down_revision = "merge_two_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("users", "is_superuser")


def downgrade() -> None:
    op.add_column("users", sa.Column("is_superuser", sa.Boolean(), nullable=True, server_default="f"))
