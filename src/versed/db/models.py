import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIM = 768


class Base(DeclarativeBase):
    pass


class Chunk(Base):
    __tablename__ = "chunk"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    version: Mapped[str] = mapped_column(String(32), index=True)
    source_path: Mapped[str] = mapped_column(Text)
    heading_path: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    has_code: Mapped[bool] = mapped_column(default=False)
    symbols_mentioned: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    token_count: Mapped[int] = mapped_column(default=0)

    __table_args__ = (Index("ix_chunk_version_source", "version", "source_path"),)


class Symbol(Base):
    __tablename__ = "symbol"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    version: Mapped[str] = mapped_column(String(32), index=True)
    qualified_name: Mapped[str] = mapped_column(Text, index=True)
    kind: Mapped[str] = mapped_column(String(32))
    signature: Mapped[str] = mapped_column(Text)
    docstring: Mapped[str | None] = mapped_column(Text, nullable=True)
    deprecated_since: Mapped[str | None] = mapped_column(String(32), nullable=True)
    alternative: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (UniqueConstraint("version", "qualified_name", name="uq_symbol_version_name"),)


class SymbolEvent(Base):
    __tablename__ = "symbol_event"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    qualified_name: Mapped[str] = mapped_column(Text, index=True)
    from_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    to_version: Mapped[str] = mapped_column(String(32))
    event_type: Mapped[str] = mapped_column(String(16))
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)


class EvalQuestion(Base):
    __tablename__ = "eval_question"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    question: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(32))
    target_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    expected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_symbols: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    should_abstain: Mapped[bool] = mapped_column(default=False)
    human_label: Mapped[str | None] = mapped_column(Text, nullable=True)
