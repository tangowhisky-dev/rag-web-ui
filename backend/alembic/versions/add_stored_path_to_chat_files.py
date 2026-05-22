"""add stored_path to chat_files

Revision ID: a1b2c3d4e5f7
Revises: f1a2b3c4d5e6
Create Date: 2026-05-22
"""
from alembic import op
import sqlalchemy as sa

revision = 'a1b2c3d4e5f7'
down_revision = 'f1a2b3c4d5e6'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('chat_files', sa.Column('stored_path', sa.String(512), nullable=True))


def downgrade():
    op.drop_column('chat_files', 'stored_path')
