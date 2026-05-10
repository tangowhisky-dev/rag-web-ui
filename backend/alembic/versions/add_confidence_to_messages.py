"""add confidence columns to messages

Revision ID: c1d2e3f4a5b6
Revises: b3c4d5e6f7a8
Create Date: 2026-05-10

Adds three columns to `messages` (assistant messages only):
  confidence_level     — very_high | high | medium | low | none | NULL
  confidence_score     — integer 0-100 | NULL
  confidence_breakdown — JSON blob with per-signal breakdown | NULL
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import LONGTEXT

revision = "c1d2e3f4a5b6"
down_revision = "b3c4d5e6f7a8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("confidence_level", sa.String(20), nullable=True))
    op.add_column("messages", sa.Column("confidence_score", sa.Integer(), nullable=True))
    op.add_column("messages", sa.Column("confidence_breakdown", LONGTEXT(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "confidence_breakdown")
    op.drop_column("messages", "confidence_score")
    op.drop_column("messages", "confidence_level")
