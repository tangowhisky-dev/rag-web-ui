"""proper_chunk_columns

Add dedicated chunk_text and chunk_index columns to document_chunks,
build a MySQL InnoDB FULLTEXT index on chunk_text, backfill from the
existing chunk_metadata JSON, then remove page_content from the JSON.

Replaces the in-memory rank_bm25 rebuild with a persistent MySQL FTS index.

Revision ID: a1b2c3d4e5f6
Revises: 5be054bd6587
Create Date: 2026-04-18 00:00:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '5be054bd6587'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add chunk_text and chunk_index columns
    op.add_column('document_chunks',
        sa.Column('chunk_text', mysql.LONGTEXT(), nullable=True))
    op.add_column('document_chunks',
        sa.Column('chunk_index', sa.Integer(), nullable=True))

    # 2. Backfill chunk_text from chunk_metadata->page_content
    op.execute("""
        UPDATE document_chunks
        SET chunk_text = JSON_UNQUOTE(JSON_EXTRACT(chunk_metadata, '$.page_content'))
        WHERE JSON_EXTRACT(chunk_metadata, '$.page_content') IS NOT NULL
    """)

    # 3. Make chunk_text NOT NULL now that backfill is done
    op.alter_column('document_chunks', 'chunk_text',
        existing_type=mysql.LONGTEXT(),
        nullable=False)

    # 4. Build FULLTEXT index on chunk_text.
    #    innodb_ft_min_token_size defaults to 3, which silently drops short
    #    tokens like "AI", "ML", "ID". Set it to 1 so every token is indexed.
    #    This requires a MySQL server variable change — handled in docker-compose
    #    via command: --innodb_ft_min_token_size=1
    #    The index itself:
    op.execute("""
        ALTER TABLE document_chunks
        ADD FULLTEXT INDEX ft_chunk_text (chunk_text)
    """)

    # 5. Remove page_content from chunk_metadata JSON (no longer needed there)
    op.execute("""
        UPDATE document_chunks
        SET chunk_metadata = JSON_REMOVE(chunk_metadata, '$.page_content')
        WHERE JSON_EXTRACT(chunk_metadata, '$.page_content') IS NOT NULL
    """)

    # 6. Also clean up redundant keys that are already proper columns
    op.execute("""
        UPDATE document_chunks
        SET chunk_metadata = JSON_REMOVE(
            JSON_REMOVE(
                JSON_REMOVE(chunk_metadata, '$.kb_id'),
                '$.document_id'
            ),
            '$.chunk_id'
        )
        WHERE chunk_metadata IS NOT NULL
    """)


def downgrade() -> None:
    # Restore page_content into JSON from chunk_text before dropping column
    op.execute("""
        UPDATE document_chunks
        SET chunk_metadata = JSON_SET(chunk_metadata, '$.page_content', chunk_text)
        WHERE chunk_text IS NOT NULL
    """)
    op.execute("ALTER TABLE document_chunks DROP INDEX ft_chunk_text")
    op.drop_column('document_chunks', 'chunk_index')
    op.drop_column('document_chunks', 'chunk_text')
