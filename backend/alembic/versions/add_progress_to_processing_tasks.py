"""add progress to processing_tasks

Revision ID: d2e3f4a5b6c7
Revises: c1d2e3f4a5b6
Create Date: 2026-05-10
"""
from alembic import op
import sqlalchemy as sa

revision = "d2e3f4a5b6c7"
down_revision = "c1d2e3f4a5b6"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("processing_tasks", sa.Column("progress", sa.Integer(), nullable=True, server_default="0"))
    op.add_column("processing_tasks", sa.Column("progress_message", sa.String(255), nullable=True))


def downgrade():
    op.drop_column("processing_tasks", "progress_message")
    op.drop_column("processing_tasks", "progress")
