"""add organisations and roles

Revision ID: add_organisations_and_roles
Revises: merge_two_heads
Create Date: 2026-06-02

"""
from alembic import op
import sqlalchemy as sa

revision = "add_organisations_and_roles"
down_revision = "merge_two_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create organisations table
    op.create_table(
        "organisations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "parent_id",
            sa.Integer(),
            sa.ForeignKey("organisations.id"),
            nullable=True,
        ),
        sa.Column("path", sa.String(1024), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_organisations_parent_id", "organisations", ["parent_id"])

    # Add org_id and role to users
    op.add_column(
        "users",
        sa.Column(
            "org_id",
            sa.Integer(),
            sa.ForeignKey("organisations.id"),
            nullable=True,
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "role",
            sa.Enum("user", "admin", "super_admin", name="userrole"),
            nullable=False,
            server_default="user",
        ),
    )
    op.create_index("ix_users_org_id", "users", ["org_id"])

    # Add org_id to knowledge_bases
    op.add_column(
        "knowledge_bases",
        sa.Column(
            "org_id",
            sa.Integer(),
            sa.ForeignKey("organisations.id"),
            nullable=True,
        ),
    )
    op.create_index("ix_knowledge_bases_org_id", "knowledge_bases", ["org_id"])

    # Add org_id to chats
    op.add_column(
        "chats",
        sa.Column(
            "org_id",
            sa.Integer(),
            sa.ForeignKey("organisations.id"),
            nullable=True,
        ),
    )
    op.create_index("ix_chats_org_id", "chats", ["org_id"])


def downgrade() -> None:
    # Drop chats.org_id
    op.drop_index("ix_chats_org_id", table_name="chats")
    op.drop_column("chats", "org_id")

    # Drop knowledge_bases.org_id
    op.drop_index("ix_knowledge_bases_org_id", table_name="knowledge_bases")
    op.drop_column("knowledge_bases", "org_id")

    # Drop users.org_id and users.role
    op.drop_index("ix_users_org_id", table_name="users")
    op.drop_column("users", "org_id")
    op.drop_column("users", "role")

    # Drop organisations table
    op.drop_index("ix_organisations_parent_id", table_name="organisations")
    op.drop_table("organisations")

    # Drop PostgreSQL ENUM type (no-op on MySQL)
    op.execute("DROP TYPE IF EXISTS userrole")
