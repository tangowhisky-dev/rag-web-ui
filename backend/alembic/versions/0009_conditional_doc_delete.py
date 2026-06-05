"""conditional doc delete on kb removal

Revision ID: 0009_conditional_doc_delete
Revises: be50d0c831b4
Create Date: 2026-06-05

Change documents.knowledge_base_id FK from CASCADE to SET NULL.
Also make the column nullable.
Add event listener for conditional deletion:
- Direct uploads (data_store_id IS NULL) → delete with KB
- DataStore docs (data_store_id IS NOT NULL) → set kb_id=NULL only

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0009_conditional_doc_delete'
down_revision = 'be50d0c831b4'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # First make the column nullable (MySQL requires type for alter)
    op.alter_column('documents', 'knowledge_base_id', 
                    existing_type=sa.Integer(),
                    nullable=True)
    
    # Drop existing FK constraint if it exists (ignore error if not)
    try:
        op.execute("""
            ALTER TABLE documents DROP FOREIGN KEY documents_ibfk_1
        """)
    except Exception:
        pass  # Constraint may already be dropped
    
    # Recreate with SET NULL instead of CASCADE
    op.create_foreign_key(
        'documents_ibfk_1',
        'documents', 'knowledge_bases',
        ['knowledge_base_id'], ['id'],
        ondelete='SET NULL'
    )


def downgrade() -> None:
    # Drop the new constraint
    op.drop_constraint('documents_ibfk_1', 'documents', type_='foreignkey')
    
    # Recreate with CASCADE (original behavior)
    op.create_foreign_key(
        'documents_ibfk_1',
        'documents', 'knowledge_bases',
        ['knowledge_base_id'], ['id'],
        ondelete='CASCADE'
    )
    
    # Make column NOT NULL again
    op.alter_column('documents', 'knowledge_base_id', 
                    existing_type=sa.Integer(),
                    nullable=False)
