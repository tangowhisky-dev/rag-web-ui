"""add documents.title column with FULLTEXT index

Revision ID: d5e6f7a8b9c1
Revises: c4d5e6f7a8b9
Create Date: 2026-08-26 10:00:00.000000

Adds a nullable title column to the documents table for storing extracted
document titles (from PDF metadata, first H1 heading, or cleaned filename).
Includes a FULLTEXT index for exact-search matching on document titles.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd5e6f7a8b9c1'
down_revision: Union[str, None] = 'c4d5e6f7a8b9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('documents', sa.Column('title', sa.String(length=512), nullable=True))
    op.execute("CREATE FULLTEXT INDEX idx_doc_title_fts ON documents(title)")


def downgrade() -> None:
    op.execute("DROP INDEX idx_doc_title_fts ON documents")
    op.drop_column('documents', 'title')
