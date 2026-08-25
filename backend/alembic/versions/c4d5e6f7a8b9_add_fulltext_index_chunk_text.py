"""add FULLTEXT index on document_chunks.chunk_text

Revision ID: c4d5e6f7a8b9
Revises: a8b2c3d4e5f6
Create Date: 2026-08-25 15:15:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'c4d5e6f7a8b9'
down_revision: Union[str, None] = 'a8b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE FULLTEXT INDEX idx_chunk_text_fts ON document_chunks(chunk_text)"
    )


def downgrade() -> None:
    op.execute(
        "DROP INDEX idx_chunk_text_fts ON document_chunks"
    )
