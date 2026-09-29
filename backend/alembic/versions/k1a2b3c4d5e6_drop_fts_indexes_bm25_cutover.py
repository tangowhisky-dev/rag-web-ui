"""drop MySQL FULLTEXT indexes — keyword retrieval moved to Qdrant BM25

Revision ID: k1a2b3c4d5e6
Revises: 78839807b756
Create Date: 2026-02-28 00:00:00.000000

Idempotent: checks information_schema before each DROP so the migration is
safe to run on databases where the indexes were already removed manually.

- idx_chunk_text_fts on document_chunks(chunk_text) — created by c4d5e6f7a8b9
- idx_doc_title_fts on documents(title) — created by d5e6f7a8b9c1

Both are dead weight after the BM25 cutover (keyword_search + /search now
use Qdrant server-side BM25, no MySQL MATCH AGAINST). The downgrade
recreates them for rollback.
"""
from typing import Sequence, Union

from alembic import op
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = 'k1a2b3c4d5e6'
down_revision: Union[str, None] = '78839807b756'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _index_exists(conn, table: str, index_name: str) -> bool:
    row = conn.execute(text(
        "SELECT 1 FROM information_schema.statistics "
        "WHERE table_schema = DATABASE() AND table_name = :t AND index_name = :i"
    ), {"t": table, "i": index_name}).first()
    return row is not None


def upgrade() -> None:
    conn = op.get_bind()
    if _index_exists(conn, "document_chunks", "idx_chunk_text_fts"):
        op.execute("DROP INDEX idx_chunk_text_fts ON document_chunks")
    if _index_exists(conn, "documents", "idx_doc_title_fts"):
        op.execute("DROP INDEX idx_doc_title_fts ON documents")


def downgrade() -> None:
    conn = op.get_bind()
    if not _index_exists(conn, "document_chunks", "idx_chunk_text_fts"):
        op.execute("CREATE FULLTEXT INDEX idx_chunk_text_fts ON document_chunks(chunk_text)")
    if not _index_exists(conn, "documents", "idx_doc_title_fts"):
        op.execute("CREATE FULLTEXT INDEX idx_doc_title_fts ON documents(title)")
