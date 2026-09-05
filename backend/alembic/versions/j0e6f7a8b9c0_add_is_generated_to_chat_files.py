"""add is_generated to chat_files

Revision ID: j0e6f7a8b9c0
Revises: i9d5e6f7a8b9
Create Date: 2026-09-05 15:00:00.000000

Adds a boolean column to chat_files to distinguish user-uploaded files
from agent-generated Office documents (pptx/docx/xlsx via OfficeCLI).

Backward compatible: existing rows default to False (uploaded files).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "j0e6f7a8b9c0"
down_revision: Union[str, None] = "i9d5e6f7a8b9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chat_files",
        sa.Column("is_generated", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("chat_files", "is_generated")
