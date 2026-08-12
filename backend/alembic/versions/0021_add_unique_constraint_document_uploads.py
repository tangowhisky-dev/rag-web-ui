"""add unique constraint on document_uploads (kb_id, file_name, file_hash)

Revision ID: 0021_unique_doc_uploads
Revises: 0020_drop_org_llm_config
Create Date: 2026-08-14 00:00:00.000000

Prevents duplicate DocumentUpload rows from concurrent upload requests.
The existing read-then-write duplicate check in the upload endpoint has a
race window; this constraint makes it enforceable at the DB level.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0021_unique_doc_uploads"
down_revision: str = "0020_drop_org_llm_config"
branch_labels: str | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_doc_upload_kb_name_hash",
        "document_uploads",
        ["knowledge_base_id", "file_name", "file_hash"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_doc_upload_kb_name_hash", "document_uploads", type_="unique")
