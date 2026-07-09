"""add clarification_requests and user_profiles tables

Revision ID: 0014_add_clarification_requests_and_user_profiles
Revises: 0013_add_message_citations_table
Create Date: 2026-01-01 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = '0014_add_clarification_requests_and_user_profiles'
down_revision = '0013_add_message_citations_table'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'clarification_requests',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('chat_id', sa.Integer(), nullable=False, index=True),
        sa.Column('assistant_message_id', sa.Integer(), nullable=False, index=True),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('options', sa.JSON(), nullable=True),
        sa.Column('rationale', sa.Text(), nullable=True),
        sa.Column('user_response', sa.Text(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('attempt', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('answered_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['chat_id'], ['chats.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['assistant_message_id'], ['messages.id']),
    )

    op.create_table(
        'user_profiles',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.Integer(), nullable=False, index=True),
        sa.Column('org_id', sa.Integer(), nullable=True, index=True),
        sa.Column('preferences_json', sa.Text(), nullable=True),
        sa.Column('query_patterns_json', sa.Text(), nullable=True),
        sa.Column('domain_focus_json', sa.Text(), nullable=True),
        sa.Column('communication_style', sa.String(32), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.UniqueConstraint('user_id', 'org_id', name='uq_user_profiles_user_org'),
    )


def downgrade() -> None:
    op.drop_table('user_profiles')
    op.drop_table('clarification_requests')
