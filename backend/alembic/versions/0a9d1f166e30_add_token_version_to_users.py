"""Add token_version to users

Revision ID: 0a9d1f166e30
Revises: 0015_add_final_confidence_to_messages
Create Date: 2026-07-26 19:08:46.494978

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0a9d1f166e30'
down_revision: Union[str, None] = '0015_add_final_confidence_to_messages'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'users',
        sa.Column(
            'token_version',
            sa.Integer(),
            nullable=False,
            server_default='0',
        ),
    )


def downgrade() -> None:
    op.drop_column('users', 'token_version')
