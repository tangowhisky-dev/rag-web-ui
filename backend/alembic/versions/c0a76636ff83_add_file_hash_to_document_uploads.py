"""add_file_hash_to_document_uploads

Revision ID: c0a76636ff83
Revises: 0008_add_datastores
Create Date: 2026-06-05 12:10:27.703227

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c0a76636ff83'
down_revision: Union[str, None] = '0008_add_datastores'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add missing columns (handle case where they might already exist)
    # file_hash
    try:
        op.execute("ALTER TABLE document_uploads ADD COLUMN file_hash VARCHAR(64) NOT NULL DEFAULT ''")
    except Exception:
        pass
    
    # temp_path
    try:
        op.execute("ALTER TABLE document_uploads ADD COLUMN temp_path VARCHAR(255) NOT NULL DEFAULT ''")
    except Exception:
        pass


def downgrade() -> None:
    op.execute("ALTER TABLE document_uploads DROP COLUMN file_hash")
