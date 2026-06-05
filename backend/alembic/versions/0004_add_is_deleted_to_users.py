"""add is_deleted and deleted_at columns to users table

Revision ID: 0004_add_is_deleted
Revises: 0003_drop_is_superuser
Create Date: 2026-06-04

"""
from alembic import op
import sqlalchemy as sa

revision = "0004_add_is_deleted"
down_revision = "0003_drop_is_superuser"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("is_deleted", sa.Boolean(), nullable=False, server_default=sa.text("0")))
    op.add_column("users", sa.Column("deleted_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "deleted_at")
    op.drop_column("users", "is_deleted")
