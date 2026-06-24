"""add last_recovered_at to data_stores

Adds last_recovered_at column to the data_stores table so that the
admin dashboard can display when the last recovery scan completed.

Revision ID: 0012_add_last_recovered_at_to_datastores
Revises: 0011_add_datastore_file_manifest
Create Date: 2026-06-23

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0012_add_last_recovered_at_to_datastores"
down_revision: Union[str, None] = ("0011_add_datastore_file_manifest", "e3f4a5b6c7d8")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "data_stores",
        sa.Column("last_recovered_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("data_stores", "last_recovered_at")
