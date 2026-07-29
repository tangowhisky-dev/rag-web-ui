"""add retrieval_score, drop citation_quality from messages

Revision ID: 0018_add_retrieval_score_citation_quality
Revises: 0017_enterprise_agent
Create Date: 2026-07-29 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = '0018_add_retrieval_score_citation_quality'
down_revision = '0017_enterprise_agent'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('messages', sa.Column('retrieval_score', sa.Integer(), nullable=True))
    op.drop_column('messages', 'citation_quality')


def downgrade() -> None:
    op.add_column('messages', sa.Column('citation_quality', sa.Integer(), nullable=True))
    op.drop_column('messages', 'retrieval_score')
