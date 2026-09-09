"""add eval_question table

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-09
"""

from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "eval_question",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("category", sa.String(32), nullable=False),
        sa.Column("target_version", sa.String(32), nullable=True),
        sa.Column("expected_answer", sa.Text(), nullable=True),
        sa.Column("expected_symbols", sa.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("should_abstain", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("human_label", sa.Text(), nullable=True),
    )
    op.create_index("ix_eval_question_category", "eval_question", ["category"])


def downgrade() -> None:
    op.drop_table("eval_question")
