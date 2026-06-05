"""merge migration heads

Merge the two heads: 0004_add_is_deleted and add_org_llm_config.

Revision ID: 0006_merge_heads
Revises: 0004_add_is_deleted, add_org_llm_config
Create Date: 2026-06-04

"""
from alembic import op
import sqlalchemy as sa

revision = "0006_merge_heads"
down_revision = ("0004_add_is_deleted", "add_org_llm_config")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
