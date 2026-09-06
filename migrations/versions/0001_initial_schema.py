"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-09-03
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "chunk",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("version", sa.String(32), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("heading_path", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("has_code", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("symbols_mentioned", sa.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("ix_chunk_version", "chunk", ["version"])
    op.create_index("ix_chunk_version_source", "chunk", ["version", "source_path"])

    op.create_table(
        "symbol",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("version", sa.String(32), nullable=False),
        sa.Column("qualified_name", sa.Text(), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("signature", sa.Text(), nullable=False),
        sa.Column("docstring", sa.Text(), nullable=True),
        sa.Column("deprecated_since", sa.String(32), nullable=True),
        sa.Column("alternative", sa.Text(), nullable=True),
        sa.UniqueConstraint("version", "qualified_name", name="uq_symbol_version_name"),
    )
    op.create_index("ix_symbol_version", "symbol", ["version"])
    op.create_index("ix_symbol_qualified_name", "symbol", ["qualified_name"])

    op.create_table(
        "symbol_event",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("qualified_name", sa.Text(), nullable=False),
        sa.Column("from_version", sa.String(32), nullable=True),
        sa.Column("to_version", sa.String(32), nullable=False),
        sa.Column("event_type", sa.String(16), nullable=False),
        sa.Column("detail", sa.Text(), nullable=True),
    )
    op.create_index("ix_symbol_event_qualified_name", "symbol_event", ["qualified_name"])


def downgrade() -> None:
    op.drop_table("symbol_event")
    op.drop_table("symbol")
    op.drop_table("chunk")
