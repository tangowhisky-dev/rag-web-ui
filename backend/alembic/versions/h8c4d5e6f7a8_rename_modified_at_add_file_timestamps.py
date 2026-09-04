"""rename modified_at to file_modified_at, add file_created_at and file_edited_at

Revision ID: h8c4d5e6f7a8
Revises: g7b3c4d5e6f7
Create Date: 2026-08-30 12:00:00.000000

Three changes to documents table:
1. Rename modified_at → file_modified_at (source file mtime from filesystem).
2. Add file_created_at (filesystem ctime / file creation time, nullable for old rows).
3. Add file_edited_at (when admin edits converted_markdown; content correction
   tracking, NOT a temporal versioning signal).

Retrieval sorts by COALESCE(file_modified_at, file_created_at) to determine
which document version is the latest.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'h8c4d5e6f7a8'
down_revision: Union[str, None] = 'g7b3c4d5e6f7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Rename modified_at → file_modified_at
    op.alter_column(
        'documents', 'modified_at',
        new_column_name='file_modified_at',
        existing_type=sa.DateTime(),
        existing_nullable=True,
    )
    # Drop old index, create new one
    op.drop_index('ix_documents_modified_at', table_name='documents')
    op.create_index('ix_documents_file_modified_at', 'documents', ['file_modified_at'])

    # 2. Add file_created_at (nullable — old rows don't have it)
    op.add_column('documents', sa.Column('file_created_at', sa.DateTime(), nullable=True))
    op.create_index('ix_documents_file_created_at', 'documents', ['file_created_at'])

    # 3. Add file_edited_at (nullable — set when admin edits converted_markdown)
    op.add_column('documents', sa.Column('file_edited_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column('documents', 'file_edited_at')
    op.drop_index('ix_documents_file_created_at', table_name='documents')
    op.drop_column('documents', 'file_created_at')
    op.drop_index('ix_documents_file_modified_at', table_name='documents')
    op.alter_column(
        'documents', 'file_modified_at',
        new_column_name='modified_at',
        existing_type=sa.DateTime(),
        existing_nullable=True,
    )
    op.create_index('ix_documents_modified_at', 'documents', ['modified_at'])
