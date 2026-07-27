"""add document file path datastore unique constraint

Adds a unique constraint on (file_path, data_store_id) to the documents
table to prevent duplicate DataStore documents from race conditions.

Revision ID: 0016_add_document_file_path_datastore_uq
Revises: 0a9d1f166e30
Create Date: 2026-07-27

"""

from typing import Sequence, Union

from alembic import op


revision: str = "0016_add_document_file_path_datastore_uq"
down_revision: Union[str, None] = "0a9d1f166e30"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_document_file_path_datastore",
        "documents",
        ["file_path", "data_store_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_document_file_path_datastore",
        "documents",
        type_="unique",
    )
