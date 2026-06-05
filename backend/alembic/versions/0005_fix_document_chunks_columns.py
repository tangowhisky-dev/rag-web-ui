"""fix document_chunks columns and enforce org_id NOT NULL

The document_chunks table was missing document_id, chunk_text, chunk_index,
and hash columns that the model requires. This migration adds them.

Also enforces org_id NOT NULL on users table.

Revision ID: 0005_fix_document_chunks
Revises: 0006_merge_heads
Create Date: 2026-06-04

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

revision = "0005_fix_document_chunks"
down_revision = "0006_merge_heads"
branch_labels = None
depends_on = None


def _column_exists(table_name, column_name):
    """Check if a column exists in a table."""
    conn = op.get_bind()
    result = conn.execute(
        sa.text(
            "SELECT COUNT(*) FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() "
            "AND TABLE_NAME = '{}' "
            "AND COLUMN_NAME = '{}'".format(table_name, column_name)
        )
    ).scalar()
    return result > 0


def upgrade() -> None:
    # 1. Add document_id column with FK to documents (only if not exists)
    if not _column_exists("document_chunks", "document_id"):
        op.add_column(
            "document_chunks",
            sa.Column("document_id", sa.Integer(), nullable=False),
        )
        op.create_foreign_key(
            "fk_document_chunks_document",
            "document_chunks",
            "documents",
            ["document_id"],
            ["id"],
        )

    # 2. Rename content -> chunk_text (LONGTEXT, NOT NULL)
    if _column_exists("document_chunks", "content") and not _column_exists("document_chunks", "chunk_text"):
        op.alter_column(
            "document_chunks",
            "content",
            new_column_name="chunk_text",
            existing_type=mysql.LONGTEXT(),
            nullable=False,
        )

    # 3. Add chunk_index column (only if not exists)
    if not _column_exists("document_chunks", "chunk_index"):
        op.add_column(
            "document_chunks",
            sa.Column("chunk_index", sa.Integer(), nullable=True),
        )

    # 4. Add hash column with index (only if not exists)
    if not _column_exists("document_chunks", "hash"):
        op.add_column(
            "document_chunks",
            sa.Column("hash", sa.String(64), nullable=False),
        )
        op.create_index("idx_hash", "document_chunks", ["hash"])

    # 5. Enforce org_id NOT NULL on users (only if still nullable)
    if not _column_exists("users", "org_id"):
        # Column doesn't exist at all - this shouldn't happen, but handle gracefully
        pass
    else:
        # Check if org_id is nullable
        conn = op.get_bind()
        result = conn.execute(
            sa.text(
                "SELECT IS_NULLABLE FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() "
                "AND TABLE_NAME = 'users' "
                "AND COLUMN_NAME = 'org_id'"
            )
        ).scalar()
        if result == "YES":
            op.alter_column(
                "users",
                "org_id",
                existing_type=mysql.INTEGER(),
                nullable=False,
            )


def downgrade() -> None:
    # Revert org_id to nullable
    conn = op.get_bind()
    result = conn.execute(
        sa.text(
            "SELECT IS_NULLABLE FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() "
            "AND TABLE_NAME = 'users' "
            "AND COLUMN_NAME = 'org_id'"
        )
    ).scalar()
    if result == "NO":
        op.alter_column(
            "users",
            "org_id",
            existing_type=mysql.INTEGER(),
            nullable=True,
        )

    # Drop hash index and column
    if _column_exists("document_chunks", "hash"):
        op.drop_index("idx_hash", table_name="document_chunks")
        op.drop_column("document_chunks", "hash")

    # Drop chunk_index
    if _column_exists("document_chunks", "chunk_index"):
        op.drop_column("document_chunks", "chunk_index")

    # Rename chunk_text back to content
    if _column_exists("document_chunks", "chunk_text"):
        op.alter_column(
            "document_chunks",
            "chunk_text",
            new_column_name="content",
            existing_type=mysql.LONGTEXT(),
            nullable=False,
        )

    # Drop FK and document_id column
    if _column_exists("document_chunks", "document_id"):
        op.drop_constraint("fk_document_chunks_document", "document_chunks", type_="foreignkey")
        op.drop_column("document_chunks", "document_id")
