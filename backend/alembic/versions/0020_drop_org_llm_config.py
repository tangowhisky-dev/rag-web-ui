"""drop org_llm_config table, migrate data to settings

Revision ID: 0020_drop_org_llm_config
Revises: 0019_add_settings_table
Create Date: 2026-08-13 00:00:00.000000

Migrates existing OrgLLMConfig rows (api_base, model_name, query_model) into
the unified settings table as org-level overrides, then drops the
org_llm_configs table.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import json

revision: str = '0020_drop_org_llm_config'
down_revision: Union[str, None] = '0019_add_settings_table'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 1. Copy existing OrgLLMConfig rows into the settings table ──
    conn = op.get_bind()

    rows = conn.execute(
        sa.text("SELECT org_id, api_base, model_name, query_model FROM org_llm_configs")
    ).fetchall()

    for row in rows:
        org_id = row[0]
        api_base = row[1]
        model_name = row[2]
        query_model = row[3]

        # Insert OPENAI_API_BASE override
        if api_base:
            conn.execute(
                sa.text("""
                    INSERT INTO settings (key, scope, org_id, value)
                    VALUES ('OPENAI_API_BASE', 'org', :org_id, :value)
                    ON DUPLICATE KEY UPDATE value = VALUES(value)
                """),
                {"org_id": org_id, "value": json.dumps(api_base)},
            )

        # Insert OPENAI_MODEL override
        if model_name:
            conn.execute(
                sa.text("""
                    INSERT INTO settings (key, scope, org_id, value)
                    VALUES ('OPENAI_MODEL', 'org', :org_id, :value)
                    ON DUPLICATE KEY UPDATE value = VALUES(value)
                """),
                {"org_id": org_id, "value": json.dumps(model_name)},
            )

        # Insert QUERY_MODEL override
        if query_model:
            conn.execute(
                sa.text("""
                    INSERT INTO settings (key, scope, org_id, value)
                    VALUES ('QUERY_MODEL', 'org', :org_id, :value)
                    ON DUPLICATE KEY UPDATE value = VALUES(value)
                """),
                {"org_id": org_id, "value": json.dumps(query_model)},
            )

    # ── 2. Drop the org_llm_configs table ──
    op.drop_table('org_llm_configs')


def downgrade() -> None:
    # Recreate the table (data is lost — this is a one-way migration)
    op.create_table(
        'org_llm_configs',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('org_id', sa.Integer(), sa.ForeignKey('organisations.id'), unique=True, nullable=False),
        sa.Column('api_base', sa.String(512), nullable=True),
        sa.Column('model_name', sa.String(255), nullable=True),
        sa.Column('query_model', sa.String(255), nullable=True),
    )
