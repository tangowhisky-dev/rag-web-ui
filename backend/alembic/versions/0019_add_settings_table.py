"""add settings table

Revision ID: 0019_add_settings_table
Revises: 0018_add_retrieval_score_citation_quality
Create Date: 2026-08-12 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0019_add_settings_table'
down_revision: Union[str, None] = '0018_add_retrieval_score_citation_quality'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'settings',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('key', sa.String(length=128), nullable=False, index=True),
        sa.Column('scope', sa.String(length=8), nullable=False),
        sa.Column('org_id', sa.Integer(), nullable=True),
        sa.Column('value', sa.Text(), nullable=True),
        sa.Column('updated_by', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['org_id'], ['organisations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['updated_by'], ['users.id'], ondelete='SET NULL'),
    )
    op.create_index('uq_org_key', 'settings', ['org_id', 'key'], unique=True)
    op.create_index('idx_settings_scope', 'settings', ['scope'])


def downgrade() -> None:
    op.drop_index('idx_settings_scope', table_name='settings')
    op.drop_index('uq_org_key', table_name='settings')
    op.drop_table('settings')
