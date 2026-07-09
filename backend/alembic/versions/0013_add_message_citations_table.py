"""add message_citations table

Revision ID: 0013_add_message_citations_table
Revises: 0012_add_last_recovered_at_to_datastores
Create Date: 2025-01-01 00:00:00.000000

Links retrieved document citations to chat messages via FK to document_chunks,
replacing the old base64-encoded context JSON stored in messages.content.
"""
from alembic import op
import sqlalchemy as sa

revision = '0013_add_message_citations_table'
down_revision = '0012_add_last_recovered_at_to_datastores'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'message_citations',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('message_id', sa.Integer(), nullable=False, index=True),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('citation_index', sa.Integer(), nullable=False),
        sa.Column('citation_metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['message_id'], ['messages.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
    )


def downgrade() -> None:
    op.drop_table('message_citations')
