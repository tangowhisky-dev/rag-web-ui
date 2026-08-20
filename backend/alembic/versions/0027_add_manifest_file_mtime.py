"""Add file_mtime column to data_store_file_manifests for incremental scanning.

Revision ID: 0027_add_manifest_file_mtime
Revises: 0026_add_production_indexes
Create Date: 2026-08-20

Stores st_mtime_ns (nanosecond precision) from os.stat() so the discovery
engine can skip hashing unchanged files.  Nullable so existing rows don't
break before the next scan populates the real value.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0027_add_manifest_file_mtime'
down_revision = '0026_add_production_indexes'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'data_store_file_manifests',
        sa.Column('file_mtime', sa.BigInteger(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('data_store_file_manifests', 'file_mtime')
