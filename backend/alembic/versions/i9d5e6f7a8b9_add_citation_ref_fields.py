"""add citation_ref fields to message_citations

Revision ID: i9d5e6f7a8b9
Revises: h8c4d5e6f7a8
Create Date: 2026-09-01 12:00:00.000000

Adds nullable columns to message_citations for the new CitationRef schema:
- citation_kind: chunk|file|section|range|grep|table|outline
- section, start_char, end_char, start_line, end_line, page, match_line, source_tool

Old rows have NULL for all new columns. The chunk_index column is altered
to be nullable=False with default=0 so non-chunk citations (file, outline,
grep) can store 0 instead of a meaningless chunk index.

Backward compatible: the frontend and backend handle NULL citation_kind
by falling back to "chunk" behavior.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'i9d5e6f7a8b9'
down_revision: Union[str, None] = 'h8c4d5e6f7a8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('message_citations', sa.Column('citation_kind', sa.String(20), nullable=True))
    op.add_column('message_citations', sa.Column('section', sa.String(255), nullable=True))
    op.add_column('message_citations', sa.Column('start_char', sa.Integer(), nullable=True))
    op.add_column('message_citations', sa.Column('end_char', sa.Integer(), nullable=True))
    op.add_column('message_citations', sa.Column('start_line', sa.Integer(), nullable=True))
    op.add_column('message_citations', sa.Column('end_line', sa.Integer(), nullable=True))
    op.add_column('message_citations', sa.Column('page', sa.Integer(), nullable=True))
    op.add_column('message_citations', sa.Column('match_line', sa.Integer(), nullable=True))
    op.add_column('message_citations', sa.Column('source_tool', sa.String(50), nullable=True))

    # Backfill citation_kind for existing rows (all old citations are chunk-based)
    op.execute("UPDATE message_citations SET citation_kind = 'chunk' WHERE citation_kind IS NULL")

    # Alter chunk_index to allow default=0 for non-chunk citations
    op.alter_column(
        'message_citations', 'chunk_index',
        existing_type=sa.Integer(),
        nullable=False,
        server_default='0',
    )


def downgrade() -> None:
    op.alter_column(
        'message_citations', 'chunk_index',
        existing_type=sa.Integer(),
        nullable=False,
        server_default=None,
    )
    op.drop_column('message_citations', 'source_tool')
    op.drop_column('message_citations', 'match_line')
    op.drop_column('message_citations', 'page')
    op.drop_column('message_citations', 'end_line')
    op.drop_column('message_citations', 'start_line')
    op.drop_column('message_citations', 'end_char')
    op.drop_column('message_citations', 'start_char')
    op.drop_column('message_citations', 'section')
    op.drop_column('message_citations', 'citation_kind')
