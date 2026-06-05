"""fix FK cascade and remove is_deleted/deleted_at from users

Fixes:
- knowledge_bases.user_id → CASCADE
- documents.knowledge_base_id → CASCADE
- document_chunks.document_id → CASCADE
- chats.user_id → CASCADE (was missing)
- messages.chat_id → CASCADE (was missing)
- chat_files.chat_id → CASCADE (was missing)
- chat_files.message_id → SET NULL (was missing)
- processing_tasks.knowledge_base_id → CASCADE (was missing)
- processing_tasks.document_id → CASCADE (was missing)
- Removes is_deleted and deleted_at columns from users

Revision ID: 0007_fix_fk_cascade
Revises: 0005_fix_document_chunks
Create Date: 2026-06-04

"""
from alembic import op
import sqlalchemy as sa

revision = "0007_fix_fk_cascade"
down_revision = "0005_fix_document_chunks"
branch_labels = None
depends_on = None


def _table_has_column(table, col):
    """Check if a table has a column."""
    conn = op.get_bind()
    sql = ("SELECT COUNT(*) FROM information_schema.COLUMNS "
           "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='{}' AND COLUMN_NAME='{}'")
    result = conn.execute(sa.text(sql.format(table, col))).scalar()
    return result > 0


def _fk_exists(table, fk_name):
    """Check if a foreign key constraint exists on a table."""
    conn = op.get_bind()
    sql = ("SELECT COUNT(*) FROM information_schema.TABLE_CONSTRAINTS "
           "WHERE CONSTRAINT_SCHEMA=DATABASE() AND TABLE_NAME='{}' "
           "AND CONSTRAINT_NAME='{}' AND CONSTRAINT_TYPE='FOREIGN KEY'")
    result = conn.execute(sa.text(sql.format(table, fk_name))).scalar()
    return result > 0


def _drop_fk_if_exists(table, fk_name):
    """Drop a foreign key constraint if it exists, silently skip otherwise."""
    if _fk_exists(table, fk_name):
        conn = op.get_bind()
        conn.execute(sa.text(
            "ALTER TABLE {} DROP FOREIGN KEY {}".format(table, fk_name)
        ))


def _add_fk(table, fk_name, col, ref_table, ref_col, on_delete=None):
    """Add a foreign key constraint, silently skip if it already exists."""
    if not _fk_exists(table, fk_name):
        conn = op.get_bind()
        stmt = "ALTER TABLE {} ADD CONSTRAINT {} FOREIGN KEY ({}) REFERENCES {}({})".format(
            table, fk_name, col, ref_table, ref_col
        )
        if on_delete:
            stmt += " ON DELETE {}".format(on_delete)
        conn.execute(sa.text(stmt))


def upgrade() -> None:
    # 1. Fix knowledge_bases.user_id → CASCADE
    _drop_fk_if_exists("knowledge_bases", "knowledge_bases_ibfk_1")
    _add_fk("knowledge_bases", "knowledge_bases_ibfk_1", "user_id", "users", "id", "CASCADE")

    # 2. Fix documents.knowledge_base_id → CASCADE
    _drop_fk_if_exists("documents", "documents_ibfk_1")
    _add_fk("documents", "documents_ibfk_1", "knowledge_base_id", "knowledge_bases", "id", "CASCADE")

    # 3. Fix document_chunks.document_id → CASCADE
    _drop_fk_if_exists("document_chunks", "fk_document_chunks_document")
    _add_fk("document_chunks", "fk_document_chunks_document", "document_id", "documents", "id", "CASCADE")

    # 4. Add missing FK: chats.user_id → CASCADE
    _add_fk("chats", "chats_ibfk_1", "user_id", "users", "id", "CASCADE")

    # 5. Add missing FK: messages.chat_id → CASCADE
    _add_fk("messages", "messages_ibfk_1", "chat_id", "chats", "id", "CASCADE")

    # 6. Add missing FK: chat_files.chat_id → CASCADE
    _add_fk("chat_files", "chat_files_ibfk_1", "chat_id", "chats", "id", "CASCADE")

    # 7. Add missing FK: chat_files.message_id → SET NULL
    _add_fk("chat_files", "chat_files_ibfk_2", "message_id", "messages", "id", "SET NULL")

    # 8. Add missing FK: processing_tasks.knowledge_base_id → CASCADE
    _add_fk("processing_tasks", "processing_tasks_ibfk_1", "knowledge_base_id", "knowledge_bases", "id", "CASCADE")

    # 9. Add missing FK: processing_tasks.document_id → CASCADE
    _add_fk("processing_tasks", "processing_tasks_ibfk_2", "document_id", "documents", "id", "CASCADE")

    # 10. Remove is_deleted and deleted_at from users (idempotent)
    for col in ("is_deleted", "deleted_at"):
        if _table_has_column("users", col):
            conn = op.get_bind()
            conn.execute(sa.text("ALTER TABLE users DROP COLUMN {}".format(col)))


def downgrade() -> None:
    # Re-add is_deleted and deleted_at (idempotent)
    for col in ("deleted_at", "is_deleted"):
        if not _table_has_column("users", col):
            conn = op.get_bind()
            if col == "deleted_at":
                conn.execute(sa.text("ALTER TABLE users ADD COLUMN deleted_at DATETIME NULL AFTER is_active"))
            else:
                conn.execute(sa.text("ALTER TABLE users ADD COLUMN is_deleted TINYINT(1) NOT NULL DEFAULT 0 AFTER deleted_at"))

    # Drop the new FKs (idempotent)
    _drop_fk_if_exists("processing_tasks", "processing_tasks_ibfk_2")
    _drop_fk_if_exists("processing_tasks", "processing_tasks_ibfk_1")
    _drop_fk_if_exists("chat_files", "chat_files_ibfk_2")
    _drop_fk_if_exists("chat_files", "chat_files_ibfk_1")
    _drop_fk_if_exists("messages", "messages_ibfk_1")
    _drop_fk_if_exists("chats", "chats_ibfk_1")
    _drop_fk_if_exists("document_chunks", "fk_document_chunks_document")
    _drop_fk_if_exists("documents", "documents_ibfk_1")
    _drop_fk_if_exists("knowledge_bases", "knowledge_bases_ibfk_1")

    # Restore original NO ACTION FKs (idempotent)
    _add_fk("document_chunks", "fk_document_chunks_document", "document_id", "documents", "id")
    _add_fk("documents", "documents_ibfk_1", "knowledge_base_id", "knowledge_bases", "id")
    _add_fk("knowledge_bases", "knowledge_bases_ibfk_1", "user_id", "users", "id")
