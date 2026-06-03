"""add SMB fields to organisations

Revision ID: 0002_add_smb_fields
Revises: 0001_add_watch_dir
Create Date: 2026-06-03 06:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '0002_add_smb_fields'
down_revision: Union[str, None] = '0001_add_watch_dir'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'organisations',
        sa.Column('smb_host', sa.String(length=255), nullable=True),
    )
    op.add_column(
        'organisations',
        sa.Column('smb_share', sa.String(length=255), nullable=True),
    )
    op.add_column(
        'organisations',
        sa.Column('smb_username', sa.String(length=255), nullable=True),
    )
    op.add_column(
        'organisations',
        sa.Column('smb_password_encrypted', sa.Text(), nullable=True),
    )
    op.add_column(
        'organisations',
        sa.Column('smb_domain', sa.String(length=255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('organisations', 'smb_domain')
    op.drop_column('organisations', 'smb_password_encrypted')
    op.drop_column('organisations', 'smb_username')
    op.drop_column('organisations', 'smb_share')
    op.drop_column('organisations', 'smb_host')
