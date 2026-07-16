"""add final answer evaluation columns to messages

Revision ID: 0015_add_final_confidence_to_messages
Revises: e4f5a6b7c8d9
Create Date: 2026-07-16 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = '0015_add_final_confidence_to_messages'
down_revision = 'e4f5a6b7c8d9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('messages', sa.Column('final_confidence', sa.Float(), nullable=True))
    op.add_column('messages', sa.Column('final_confidence_level', sa.String(20), nullable=True))
    op.add_column('messages', sa.Column('faithfulness', sa.Integer(), nullable=True))
    op.add_column('messages', sa.Column('completeness', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('messages', 'completeness')
    op.drop_column('messages', 'faithfulness')
    op.drop_column('messages', 'final_confidence_level')
    op.drop_column('messages', 'final_confidence')
