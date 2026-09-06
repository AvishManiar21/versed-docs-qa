"""embedding dim 1536 -> 768 (switch to local Ollama nomic-embed-text)

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-06
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE chunk ALTER COLUMN embedding TYPE vector(768)")


def downgrade() -> None:
    op.execute("ALTER TABLE chunk ALTER COLUMN embedding TYPE vector(1536)")
