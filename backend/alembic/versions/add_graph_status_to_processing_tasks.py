"""add graph_status and graph_error to processing_tasks

Revision ID: e3f4a5b6c7d8
Revises: d2e3f4a5b6c7
Create Date: 2026-05-16
"""
from alembic import op
import sqlalchemy as sa

revision = "e3f4a5b6c7d8"
down_revision = "d2e3f4a5b6c7"
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()

    result = conn.execute(sa.text(
        "SELECT COUNT(*) FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'processing_tasks' AND COLUMN_NAME = 'graph_status'"
    ))
    if result.scalar() == 0:
        op.add_column("processing_tasks", sa.Column("graph_status", sa.String(50), nullable=True))

    result = conn.execute(sa.text(
        "SELECT COUNT(*) FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'processing_tasks' AND COLUMN_NAME = 'graph_error'"
    ))
    if result.scalar() == 0:
        op.add_column("processing_tasks", sa.Column("graph_error", sa.Text(), nullable=True))


def downgrade():
    op.drop_column("processing_tasks", "graph_error")
    op.drop_column("processing_tasks", "graph_status")
