# Versed Foundation (Corpus Ingestion & Symbol Timeline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the two data sources Versed's whole system depends on — a
version-tagged chunk corpus of LangChain's documentation, and a symbol
timeline (added/changed/deprecated/moved/removed) built by introspecting real
installed package versions — queryable end-to-end from the CLI.

**Architecture:** A Python project (`uv`-managed) with a Postgres + pgvector
store. Two independent ingestion pipelines write into that store: a docs
pipeline (git checkout → structural chunk → embed → persist) and a symbol
pipeline (isolated per-version venv → introspect → persist → diff into
events). A thin Typer CLI drives both and exposes a `timeline <symbol>` query
— this plan's demo deliverable.

**Tech Stack:** Python 3.12, `uv`, SQLAlchemy 2.0 + `psycopg` v3, `pgvector`,
Alembic, `pydantic-settings`, `langchain-core`/`langchain-ollama`, `typer`,
`tiktoken`, `httpx`, Postgres 16 + pgvector (Docker), Ollama (local
embeddings), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-09-02-version-aware-docs-qa-design.md`
(covers Milestones 1–2: Corpus, Symbol timeline)

## Global Constraints

- Python 3.12 for the main project; Python 3.11 for every per-version
  introspection venv (compatible with LangChain 0.1 through current).
- `uv` is the only package/venv manager used anywhere in this plan — it
  bundles its own pip and can fetch interpreters, which avoids the
  system-`ensurepip` dependency that plain `venv` has.
- Embeddings: local Ollama `nomic-embed-text` (768 dimensions) via
  `langchain-ollama`. Requires the Ollama daemon running locally
  (`systemctl is-active ollama`) with the model pulled
  (`ollama pull nomic-embed-text`) — no API key, no cost, no external
  network call for any step that touches `ingest_docs_for_version`. Changed
  from an earlier OpenAI-based design (see Task 6B) because the user has no
  OpenAI budget; verified live (not assumed) that the model returns 768-dim
  vectors before committing to this dimension in the schema.
- Postgres runs via Docker Compose on host port **5433** (not 5432, to avoid
  clashing with a local Postgres install). `DATABASE_URL` in `.env` points
  there.
- The `scripts/introspect_worker.py` script is stdlib-only — **zero**
  third-party imports. It runs inside an isolated venv that has only the
  target LangChain version installed, not this project's own dependencies.
  This is load-bearing: any edit that adds a `versed.*` import to that file
  will break every version's introspection.
- **Prefer a library over hand-rolled code whenever one genuinely fits.**
  Every task review checks this explicitly, not just correctness — e.g. use
  `tiktoken` rather than a hand-rolled token estimator (already done),
  `httpx` rather than raw `urllib` (already done), `pip-audit`/CodeQL rather
  than writing vulnerability detection (Task 14), `ruff format` rather than
  a bespoke style checker. The one deliberate exception is
  `scripts/introspect_worker.py` (Task 8): it must stay stdlib-only by
  design, since it runs inside an isolated venv that only has the target
  library installed — a library dependency there would defeat the point.
- **Known scope limit (carried forward, not hidden):** this plan introspects
  only the `langchain` package — **update (Task 12B): the current version now
  also introspects `langchain-classic`, so `langchain` → `langchain_classic`
  relocations (e.g. `ContextualCompressionRetriever`) now correctly surface
  as a `moved` event instead of a bare `removed` one.** The
  `create_react_agent` (langgraph.prebuilt)
  → `create_agent` (langchain.agents) rename spans two packages and will not
  appear as a `moved` event until a later plan extends the manifest to also
  introspect `langgraph` per version point. **Second, independent reason
  that specific rename still wouldn't surface as `moved` even after that
  extension:** Task 10's move-pairing heuristic matches a removed and an
  added symbol only on exact short-name equality (`create_react_agent` !=
  `create_agent` — the name itself changed, not just the module path). It
  would correctly surface as a separate `removed` + `added` pair instead —
  accurate, just not merged into one `moved` event. The heuristic is
  intentionally this simple (no fuzzy/similarity matching) per the
  library-preference and YAGNI rules below; Task 10's own test uses a
  genuine LangChain example where the short name IS preserved
  (`ContextualCompressionRetriever` moving from `langchain.retrievers` to
  `langchain_classic.retrievers.contextual_compression` in the 1.0
  migration) to test the mechanism correctly.
- **Deferred production-grade techniques, tracked rather than forgotten.**
  Not needed at this plan's stage; flagged with the specific point each
  becomes relevant so a later task doesn't silently skip it:
  - *Structured logging* — irrelevant while everything is a CLI script.
    Add when the FastAPI service exists (spec Milestone 8, a later plan),
    not to this plan's ingestion code.
  - *Retry logic on external calls* — Task 7's OpenAI embedding calls and
    Task 9's per-version `pip install` both hit real network services with
    no retry; `check=True` on subprocess calls means a failure is loud
    rather than silent, which is the correct default for now, but a
    transient network blip currently kills the whole ingestion run. When
    Tasks 7 and 9 are implemented, prefer a library (`tenacity`) over a
    hand-rolled retry loop, per the library-preference rule above — don't
    add this preemptively before those tasks exist.
  - *Full error-handling/observability layer* (typed exceptions,
    correlation IDs, etc.) — belongs to the service layer once one exists,
    not this plan's ingestion CLI.
- Concrete version manifest (verified against the real repos before writing
  this plan — see rationale below):

  | version | docs source | pip spec |
  |---|---|---|
  | `0.1` | `langchain-ai/langchain@v0.1.0`, subpath `docs/docs` | `langchain==0.1.0` |
  | `0.2` | `langchain-ai/langchain@langchain==0.2.0`, subpath `docs/docs` | `langchain==0.2.0` |
  | `0.3` | `langchain-ai/langchain@langchain==0.3.0`, subpath `docs/docs` | `langchain==0.3.0` |
  | `<resolved latest>` | `langchain-ai/docs@main`, subpath `src/oss/python` | `langchain==<resolved latest>` |

  Rationale: `langchain-ai/docs` (the current Mintlify docs site) was created
  May 2025 and carries no version tags — it only reflects current docs on
  `main`. There is no way to check out "docs as of exactly 1.0.0" from it. The
  spec's original 5-version plan (0.1/0.2/0.3/1.0/current) is reduced to 4
  concrete points (0.1/0.2/0.3/current) for this reason — every breaking
  change this project cares about (the 0.3→1.0 API renames) is still fully
  captured, since "current" is already past 1.0. The 4th version's git tag is
  `v0.1.0` (early tags used a `v` prefix); `0.2.0`/`0.3.0` use the later
  `langchain==X.Y.Z` tag scheme — both verified live against the GitHub API.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `README.md`

**Interfaces:**
- Produces: a `uv`-managed project at the repo root that `uv sync` succeeds
  against, with `src/versed` as the importable package root for every later
  task.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "versed"
version = "0.1.0"
description = "Version-aware documentation QA over LangChain's docs"
requires-python = ">=3.12"
dependencies = [
    "sqlalchemy>=2.0.35",
    "psycopg[binary]>=3.2",
    "pgvector>=0.3.6",
    "alembic>=1.13.3",
    "pydantic-settings>=2.5",
    "langchain-core>=1.0.0",
    "langchain-openai>=1.0.0",
    "typer>=0.12.5",
    "tiktoken>=0.8.0",
    "httpx>=0.27",
]

[dependency-groups]
dev = [
    "pytest>=8.3",
    "pytest-cov>=5.0",
    "ruff>=0.6",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/versed"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: requires a running Postgres (docker compose up -d db)",
    "slow: hits the network (git clone, pip install, OpenAI embeddings)",
]

[tool.ruff]
line-length = 100
src = ["src", "tests"]
```

- [ ] **Step 2: Write `.env.example`**

```
DATABASE_URL=postgresql+psycopg://versed:versed@localhost:5433/versed
OPENAI_API_KEY=sk-replace-me
```

- [ ] **Step 3: Write `.gitignore`**

```
.venv/
.versed-venvs/
__pycache__/
*.pyc
.env
.pytest_cache/
.ruff_cache/
*.egg-info/
```

- [ ] **Step 4: Write `README.md` stub**

```markdown
# Versed

Version-aware documentation QA over LangChain's docs. See
`docs/superpowers/specs/2026-09-02-version-aware-docs-qa-design.md` for the
full design.

## Setup

    cp .env.example .env   # fill in OPENAI_API_KEY
    uv sync
    docker compose up -d db
    uv run alembic upgrade head
```

- [ ] **Step 5: Create package skeleton and verify `uv sync`**

```bash
mkdir -p src/versed/db src/versed/ingest scripts tests/unit tests/integration
touch src/versed/__init__.py src/versed/db/__init__.py src/versed/ingest/__init__.py
uv sync
```

Expected: `uv sync` creates `.venv/` and `uv.lock` with no errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .env.example .gitignore README.md src tests scripts uv.lock
git commit -m "chore: scaffold versed project with uv"
```

---

### Task 1B: Pre-commit hooks

**Files:**
- Create: `.pre-commit-config.yaml`
- Modify: `pyproject.toml` (add `pre-commit` to the `dev` dependency group)
- Modify: `README.md` (document `uv run pre-commit install` as a setup step)

**Interfaces:**
- Produces: a git pre-commit hook that runs `ruff check --fix` and
  `ruff format` on staged files, blocking the commit if either finds
  something it can't auto-fix. Uses the existing `ruff` config in
  `pyproject.toml` — no separate configuration to maintain.

- [ ] **Step 1: Add `pre-commit` to `pyproject.toml`'s dev group**

```toml
    "pre-commit>=3.8",
```

- [ ] **Step 2: Write `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

- [ ] **Step 3: Install the hook and verify it blocks a real violation**

```bash
uv sync --all-groups
uv run pre-commit install
```

Then introduce a deliberate lint violation (e.g. an unused import) in a
tracked file, `git add` it, and run `git commit -m "test"`. Expected: the
commit is blocked or the file is auto-fixed by the hook (ruff's `--fix`
will silently correct what it can, then the commit re-run succeeds); revert
the deliberate violation afterward — this is a manual verification step,
not a permanent test file.

- [ ] **Step 4: Verify a clean run**

```bash
uv run pre-commit run --all-files
```

Expected: all hooks pass (exit 0).

- [ ] **Step 5: Add the setup step to README.md**

In the "Get it running" section, after `uv sync`, add:

```
uv run pre-commit install    # runs ruff automatically on every commit
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .pre-commit-config.yaml README.md
git commit -m "chore: add pre-commit hooks for ruff check and format"
```

---

### Task 2: Config and Docker Compose

**Files:**
- Create: `src/versed/config.py`
- Create: `docker-compose.yml`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `versed.config.settings` — a `Settings` instance with
  `.database_url: str`, `.openai_api_key: str`, `.data_dir: str`. Every later
  task that touches the DB or OpenAI imports this.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_config.py
import os

from versed.config import Settings


def test_settings_reads_database_url_from_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://x:y@host/db")
    settings = Settings()
    assert settings.database_url == "postgresql+psycopg://x:y@host/db"


def test_settings_has_sane_defaults():
    settings = Settings(_env_file=None)
    assert settings.database_url.startswith("postgresql+psycopg://")
    assert settings.data_dir == "data"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'versed.config'`

- [ ] **Step 3: Write `src/versed/config.py`**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://versed:versed@localhost:5433/versed"
    openai_api_key: str = ""
    data_dir: str = "data"


settings = Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write `docker-compose.yml`**

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: versed
      POSTGRES_PASSWORD: versed
      POSTGRES_DB: versed
    ports:
      - "5433:5432"
    volumes:
      - versed_pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U versed"]
      interval: 5s
      timeout: 3s
      retries: 10

volumes:
  versed_pgdata:
```

- [ ] **Step 6: Verify Postgres starts**

```bash
docker compose up -d db
docker compose ps
```

Expected: `db` service shows `healthy` within ~15 seconds.

- [ ] **Step 7: Commit**

```bash
git add src/versed/config.py docker-compose.yml tests/unit/test_config.py
git commit -m "feat: add settings and dockerized Postgres+pgvector"
```

---

### Task 3: Database models and initial migration

**Files:**
- Create: `src/versed/db/models.py`
- Create: `src/versed/db/session.py`
- Create: `alembic.ini`
- Create: `migrations/env.py`
- Create: `migrations/script.py.mako`
- Create: `migrations/versions/0001_initial_schema.py`
- Test: `tests/integration/test_schema.py`

**Interfaces:**
- Consumes: `versed.config.settings` (Task 2)
- Produces: `Base`, `Chunk`, `Symbol`, `SymbolEvent` from
  `versed.db.models`; `get_session()` context manager from
  `versed.db.session` yielding a `sqlalchemy.orm.Session`. Every later
  ingestion/query task uses these exact names.

- [ ] **Step 1: Write `src/versed/db/models.py`**

```python
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIM = 1536


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

    __table_args__ = (
        UniqueConstraint("version", "qualified_name", name="uq_symbol_version_name"),
    )


class SymbolEvent(Base):
    __tablename__ = "symbol_event"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    qualified_name: Mapped[str] = mapped_column(Text, index=True)
    from_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    to_version: Mapped[str] = mapped_column(String(32))
    event_type: Mapped[str] = mapped_column(String(16))
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
```

- [ ] **Step 2: Write `src/versed/db/session.py`**

```python
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from versed.config import settings

engine = create_engine(settings.database_url, echo=False)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

- [ ] **Step 3: Initialize Alembic**

```bash
uv run alembic init migrations
```

- [ ] **Step 4: Edit `migrations/env.py`**

Replace the `target_metadata = None` line and the URL wiring near the top
with:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from versed.config import settings  # noqa: E402
from versed.db.models import Base  # noqa: E402

config.set_main_option("sqlalchemy.url", settings.database_url)
target_metadata = Base.metadata
```

(Insert this block after the existing `config = context.config` line that
Alembic's generated `env.py` already contains; leave the rest of the
generated file as-is.)

- [ ] **Step 5: Write `migrations/versions/0001_initial_schema.py`**

```python
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
```

- [ ] **Step 6: Apply the migration and verify**

```bash
docker compose up -d db
uv run alembic upgrade head
```

Expected: no errors; `alembic_version` table and the three tables exist.

- [ ] **Step 7: Write the integration test**

```python
# tests/integration/test_schema.py
import pytest
from sqlalchemy import inspect

from versed.db.session import engine

pytestmark = pytest.mark.integration


def test_tables_exist():
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert {"chunk", "symbol", "symbol_event"} <= tables
```

- [ ] **Step 8: Run it**

Run: `uv run pytest tests/integration/test_schema.py -v -m integration`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/versed/db alembic.ini migrations tests/integration/test_schema.py
git commit -m "feat: add DB models and initial Alembic migration"
```

---

### Task 4: Version manifest

**Files:**
- Create: `src/versed/ingest/manifest.py`
- Test: `tests/unit/test_manifest.py`

**Interfaces:**
- Produces: `VersionSource` dataclass (`version`, `repo_url`, `ref`,
  `docs_subpath`, `pip_spec`), `resolve_latest_langchain_version() -> str`,
  `build_manifest() -> list[VersionSource]`. Every ingestion and CLI task
  iterates `build_manifest()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_manifest.py
import httpx

from versed.ingest.manifest import build_manifest, resolve_latest_langchain_version


class _FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"info": {"version": "1.9.9"}}


def test_resolve_latest_langchain_version(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse())
    assert resolve_latest_langchain_version() == "1.9.9"


def test_build_manifest_has_four_versions(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse())
    manifest = build_manifest()
    assert [s.version for s in manifest] == ["0.1", "0.2", "0.3", "1.9.9"]
    assert manifest[0].ref == "v0.1.0"
    assert manifest[2].ref == "langchain==0.3.0"
    assert manifest[3].repo_url == "https://github.com/langchain-ai/docs.git"
    assert manifest[3].pip_spec == "langchain==1.9.9"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'versed.ingest.manifest'`

- [ ] **Step 3: Write `src/versed/ingest/manifest.py`**

```python
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class VersionSource:
    version: str
    repo_url: str
    ref: str
    docs_subpath: str
    pip_spec: str


def resolve_latest_langchain_version() -> str:
    response = httpx.get("https://pypi.org/pypi/langchain/json", timeout=10.0)
    response.raise_for_status()
    return response.json()["info"]["version"]


def build_manifest() -> list[VersionSource]:
    latest = resolve_latest_langchain_version()
    langchain_repo = "https://github.com/langchain-ai/langchain.git"
    return [
        VersionSource(
            version="0.1",
            repo_url=langchain_repo,
            ref="v0.1.0",
            docs_subpath="docs/docs",
            pip_spec="langchain==0.1.0",
        ),
        VersionSource(
            version="0.2",
            repo_url=langchain_repo,
            ref="langchain==0.2.0",
            docs_subpath="docs/docs",
            pip_spec="langchain==0.2.0",
        ),
        VersionSource(
            version="0.3",
            repo_url=langchain_repo,
            ref="langchain==0.3.0",
            docs_subpath="docs/docs",
            pip_spec="langchain==0.3.0",
        ),
        VersionSource(
            version=latest,
            repo_url="https://github.com/langchain-ai/docs.git",
            ref="main",
            docs_subpath="src/oss/python",
            pip_spec=f"langchain=={latest}",
        ),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_manifest.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/versed/ingest/manifest.py tests/unit/test_manifest.py
git commit -m "feat: add version manifest with resolved latest LangChain version"
```

---

### Task 5: Structural chunker and text features

**Files:**
- Create: `src/versed/ingest/chunker.py`
- Create: `src/versed/ingest/text_features.py`
- Test: `tests/unit/test_chunker.py`
- Test: `tests/unit/test_text_features.py`

**Interfaces:**
- Produces: `RawChunk` dataclass (`heading_path: list[str]`, `content: str`,
  `has_code: bool`), `chunk_markdown(text: str) -> list[RawChunk]` from
  `versed.ingest.chunker`; `count_tokens(content: str) -> int` and
  `extract_symbols_mentioned(content: str) -> list[str]` from
  `versed.ingest.text_features`. Task 7 (docs pipeline) consumes both.

- [ ] **Step 1: Write the failing chunker tests**

```python
# tests/unit/test_chunker.py
from versed.ingest.chunker import chunk_markdown


def test_splits_on_headings_and_tracks_path():
    text = """# Agents

Intro text.

## Create an agent

Use `create_agent`.

## Migrating

Old code moved.
"""
    chunks = chunk_markdown(text)
    assert [c.heading_path for c in chunks] == [
        ["Agents"],
        ["Agents", "Create an agent"],
        ["Agents", "Migrating"],
    ]
    assert "Use `create_agent`." in chunks[1].content


def test_never_splits_inside_fenced_code_block():
    text = """# Title

```python
# This looks like a heading but is inside a fence
def f():
    pass
```

After code.
"""
    chunks = chunk_markdown(text)
    assert len(chunks) == 1
    assert "def f():" in chunks[0].content
    assert chunks[0].has_code is True


def test_no_heading_produces_single_unheaded_chunk():
    chunks = chunk_markdown("Just a paragraph, no headings.")
    assert len(chunks) == 1
    assert chunks[0].heading_path == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_chunker.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `src/versed/ingest/chunker.py`**

```python
import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_RE = re.compile(r"^(```|~~~)")


@dataclass
class RawChunk:
    heading_path: list[str]
    content: str
    has_code: bool


def chunk_markdown(text: str) -> list[RawChunk]:
    """Split markdown/MDX into chunks on heading boundaries.

    Never splits inside a fenced code block. Each chunk's heading_path is
    the full stack of headings above it.
    """
    lines = text.splitlines()
    chunks: list[RawChunk] = []

    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        content = "\n".join(current_lines).strip()
        if content:
            chunks.append(
                RawChunk(
                    heading_path=[title for _, title in heading_stack],
                    content=content,
                    has_code="```" in content or "~~~" in content,
                )
            )
        current_lines.clear()

    for line in lines:
        stripped = line.strip()
        fence_match = _FENCE_RE.match(stripped)
        if fence_match:
            if not in_fence:
                in_fence = True
                fence_marker = fence_match.group(1)
            elif stripped.startswith(fence_marker):
                in_fence = False
            current_lines.append(line)
            continue

        if in_fence:
            current_lines.append(line)
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            continue

        current_lines.append(line)

    flush()
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_chunker.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Write the failing text-features tests**

```python
# tests/unit/test_text_features.py
from versed.ingest.text_features import count_tokens, extract_symbols_mentioned


def test_extracts_inline_code_identifiers():
    content = "Use `create_agent` instead of `create_react_agent` here."
    assert extract_symbols_mentioned(content) == ["create_agent", "create_react_agent"]


def test_ignores_non_identifier_inline_code():
    content = "Run `pip install langchain` then import it."
    assert extract_symbols_mentioned(content) == []


def test_count_tokens_is_positive_and_roughly_proportional():
    short = count_tokens("hello world")
    long = count_tokens("hello world " * 50)
    assert 0 < short < long
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_text_features.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 7: Write `src/versed/ingest/text_features.py`**

```python
import re

import tiktoken

_INLINE_CODE_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)`")
_encoding = tiktoken.get_encoding("cl100k_base")


def extract_symbols_mentioned(content: str) -> list[str]:
    return sorted(set(_INLINE_CODE_RE.findall(content)))


def count_tokens(content: str) -> int:
    return len(_encoding.encode(content))
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_text_features.py -v`
Expected: PASS (3 tests)

- [ ] **Step 9: Commit**

```bash
git add src/versed/ingest/chunker.py src/versed/ingest/text_features.py tests/unit/test_chunker.py tests/unit/test_text_features.py
git commit -m "feat: add structural markdown chunker and text features"
```

---

### Task 6: Docs fetcher (git checkout by ref)

**Files:**
- Create: `src/versed/ingest/docs_fetch.py`
- Test: `tests/unit/test_docs_fetch.py`

**Interfaces:**
- Produces: `fetch_ref(repo_url: str, ref: str, dest: Path) -> None` and
  `iter_doc_files(root: Path, subpath: str) -> list[Path]` from
  `versed.ingest.docs_fetch`. Task 7 consumes both.

- [ ] **Step 1: Write the failing test (uses a local git repo as fixture — no network)**

```python
# tests/unit/test_docs_fetch.py
import subprocess
from pathlib import Path

import pytest

from versed.ingest.docs_fetch import fetch_ref, iter_doc_files


@pytest.fixture
def local_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source_repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)

    docs_dir = repo / "docs" / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "intro.md").write_text("# Intro\n\nHello.")
    (docs_dir / "guide.mdx").write_text("# Guide\n\nHi.")
    (repo / "README.md").write_text("not in subpath")

    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    subprocess.run(["git", "tag", "v0.1.0"], cwd=repo, check=True)
    return repo


def test_fetch_ref_checks_out_tag(local_repo: Path, tmp_path: Path):
    dest = tmp_path / "checkout"
    fetch_ref(f"file://{local_repo}", "v0.1.0", dest)
    assert (dest / "docs" / "docs" / "intro.md").exists()


def test_iter_doc_files_only_returns_subpath_md_and_mdx(local_repo: Path, tmp_path: Path):
    dest = tmp_path / "checkout2"
    fetch_ref(f"file://{local_repo}", "v0.1.0", dest)
    files = iter_doc_files(dest, "docs/docs")
    names = sorted(p.name for p in files)
    assert names == ["guide.mdx", "intro.md"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_docs_fetch.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `src/versed/ingest/docs_fetch.py`**

```python
import subprocess
from pathlib import Path


def fetch_ref(repo_url: str, ref: str, dest: Path) -> None:
    """Shallow-clone a single ref (tag or branch) of repo_url into dest."""
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; remove it before fetching")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", ref, repo_url, str(dest)],
        check=True,
        capture_output=True,
        text=True,
    )


def iter_doc_files(root: Path, subpath: str) -> list[Path]:
    """Return every .md/.mdx file under root/subpath, sorted for determinism."""
    base = root / subpath
    return sorted(p for p in base.rglob("*") if p.suffix in {".md", ".mdx"})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_docs_fetch.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/versed/ingest/docs_fetch.py tests/unit/test_docs_fetch.py
git commit -m "feat: add git-ref docs fetcher"
```

---

### Task 6B: Switch embeddings to local Ollama

**Files:**
- Modify: `pyproject.toml` (remove `langchain-openai`, add `langchain-ollama`)
- Modify: `src/versed/db/models.py` (`EMBEDDING_DIM` 1536 → 768)
- Modify: `src/versed/config.py` (remove unused `openai_api_key` field)
- Modify: `.env.example`, `README.md`
- Create: `migrations/versions/0002_embedding_dim_768.py`

**Interfaces:**
- Produces: schema and dependencies aligned on `nomic-embed-text`'s real
  768-dimensional output (verified live against a running Ollama instance
  before writing this task — not assumed). No interface signatures change;
  `Chunk.embedding`'s Python type stays `list[float] | None`.
- Reason for the change: user has no OpenAI budget. Ollama runs locally,
  free, no API key. Reserving OpenRouter's free-tier chat models for a
  later plan's LLM/judge needs — a separate decision, not this one.

- [ ] **Step 1: Update `pyproject.toml`**

Remove `"langchain-openai>=1.0.0",` from `dependencies`, add:

```toml
    "langchain-ollama>=0.2.0",
```

- [ ] **Step 2: Update `EMBEDDING_DIM` in `src/versed/db/models.py`**

```python
EMBEDDING_DIM = 768
```

- [ ] **Step 3: Remove the unused `openai_api_key` field from `src/versed/config.py`**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://versed:versed@localhost:5433/versed"
    data_dir: str = "data"


settings = Settings()
```

Confirmed safe: no test in `tests/unit/test_config.py` asserts on
`openai_api_key`.

- [ ] **Step 4: Write `migrations/versions/0002_embedding_dim_768.py`**

```python
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
```

Raw SQL, not `op.alter_column`, because pgvector's dimension is a type
parameter Alembic's generic column-alter helper doesn't model — this is the
standard way to do it. Safe to run with zero rows in the table (true right
now); if it's ever run against a populated table, note that a `vector(N)`
type change is only lossless when every existing value already has exactly
N dimensions — not a concern yet since nothing has been ingested.

- [ ] **Step 5: Apply the migration and verify**

```bash
uv sync --all-groups
docker compose up -d db
uv run alembic upgrade head
docker compose exec -T db psql -U versed -d versed -c "\d chunk" | grep embedding
```

Expected: shows `embedding | vector(768)`, and `uv run alembic current`
reports `0002 (head)`.

- [ ] **Step 6: Update `.env.example`**

```
DATABASE_URL=postgresql+psycopg://versed:versed@localhost:5433/versed
```

(Drop the `OPENAI_API_KEY` line entirely — nothing in this plan needs it.)

- [ ] **Step 7: Update `README.md`**

In the prerequisites table, replace the "An OpenAI API key" row with:

| **[Ollama](https://ollama.com)** | local embeddings (`nomic-embed-text`) | `curl -fsSL https://ollama.com/install.sh \| sh` then `ollama pull nomic-embed-text` |

In "Get it running", remove the `cp .env.example .env` comment line
referencing `OPENAI_API_KEY` (the `cp` command itself stays; only the
comment about what to fill in changes, since there's nothing left to fill
in for a local Postgres+Ollama setup — delete the comment line entirely).

- [ ] **Step 8: Verify everything still passes**

```bash
uv run pytest tests/unit -v
uv run ruff check .
uv run ruff format --check .
```

Expected: all green. `tests/unit/test_config.py` still passes with the
`openai_api_key` field removed (it was never asserted on).

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml src/versed/db/models.py src/versed/config.py .env.example README.md migrations/versions/0002_embedding_dim_768.py
git commit -m "feat: switch embeddings to local Ollama (nomic-embed-text, 768-dim)"
```

---

### Task 7: Docs ingestion pipeline

**Files:**
- Create: `src/versed/ingest/docs_pipeline.py`
- Test: `tests/integration/test_docs_pipeline.py`

**Interfaces:**
- Consumes: `VersionSource` (Task 4), `chunk_markdown`/`RawChunk` (Task 5),
  `count_tokens`/`extract_symbols_mentioned` (Task 5), `fetch_ref`/
  `iter_doc_files` (Task 6), `Chunk` model (Task 3), `Session`.
- Produces: `ingest_docs_for_version(source: VersionSource, session: Session) -> int`
  (returns chunk count). Task 12 (CLI) consumes this.

- [ ] **Step 1: Write `src/versed/ingest/docs_pipeline.py`**

```python
import tempfile
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from sqlalchemy.orm import Session

from versed.db.models import Chunk
from versed.ingest.chunker import chunk_markdown
from versed.ingest.docs_fetch import fetch_ref, iter_doc_files
from versed.ingest.manifest import VersionSource
from versed.ingest.text_features import count_tokens, extract_symbols_mentioned

EMBED_BATCH_SIZE = 100


def ingest_docs_for_version(source: VersionSource, session: Session) -> int:
    """Fetch, chunk, embed, and persist one version's docs. Returns chunk count."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "repo"
        fetch_ref(source.repo_url, source.ref, dest)
        doc_files = iter_doc_files(dest, source.docs_subpath)

        raw_chunks = []
        for path in doc_files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for raw in chunk_markdown(text):
                raw_chunks.append((path.relative_to(dest), raw))

        session.query(Chunk).filter_by(version=source.version).delete()

        for batch_start in range(0, len(raw_chunks), EMBED_BATCH_SIZE):
            batch = raw_chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
            vectors = embeddings.embed_documents([raw.content for _, raw in batch])
            for (rel_path, raw), vector in zip(batch, vectors):
                session.add(
                    Chunk(
                        version=source.version,
                        source_path=str(rel_path),
                        heading_path=" > ".join(raw.heading_path),
                        content=raw.content,
                        embedding=vector,
                        has_code=raw.has_code,
                        symbols_mentioned=extract_symbols_mentioned(raw.content),
                        token_count=count_tokens(raw.content),
                    )
                )

        session.commit()
        return len(raw_chunks)
```

- [ ] **Step 2: Write the integration test (real network to GitHub for the git clone + a running local Ollama daemon for embeddings — no OpenAI key needed)**

```python
# tests/integration/test_docs_pipeline.py
import pytest
from sqlalchemy import select

from versed.db.models import Chunk
from versed.db.session import get_session
from versed.ingest.docs_pipeline import ingest_docs_for_version
from versed.ingest.manifest import VersionSource

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_ingest_docs_for_version_persists_chunks():
    source = VersionSource(
        version="0.1",
        repo_url="https://github.com/langchain-ai/langchain.git",
        ref="v0.1.0",
        docs_subpath="docs/docs",
        pip_spec="langchain==0.1.0",
    )
    with get_session() as session:
        session.query(Chunk).filter_by(version="0.1").delete()
        session.commit()

        count = ingest_docs_for_version(source, session)
        assert count > 0

        stored = session.scalars(select(Chunk).where(Chunk.version == "0.1")).all()
        assert len(stored) == count
        assert all(c.embedding is not None for c in stored)
```

- [ ] **Step 3: Run it (requires `docker compose up -d db`, Ollama daemon running with `nomic-embed-text` pulled, network to GitHub)**

Run: `uv run pytest tests/integration/test_docs_pipeline.py -v -m integration`
Expected: PASS — prints a nonzero chunk count for the real 0.1.0 docs checkout

- [ ] **Step 4: Commit**

```bash
git add src/versed/ingest/docs_pipeline.py tests/integration/test_docs_pipeline.py
git commit -m "feat: add docs ingestion pipeline (fetch, chunk, embed, persist)"
```

---

### Task 8: Standalone symbol introspection worker

**Files:**
- Create: `scripts/introspect_worker.py`
- Test: `tests/unit/test_introspect_worker.py`

**Interfaces:**
- Produces: `parse_deprecation(doc: str | None) -> tuple[str | None, str | None]`
  (importable for unit testing) and a CLI entry point
  `python scripts/introspect_worker.py <package_name>` that prints one JSON
  object per line to stdout with keys `qualified_name`, `kind`, `signature`,
  `docstring`, `deprecated_since`, `alternative`. Task 9 consumes the CLI
  entry point via subprocess.
- **Zero third-party imports** — stdlib only (`inspect`, `pkgutil`,
  `importlib`, `re`, `json`, `sys`). This script is copied into isolated
  per-version venvs; it cannot depend on anything from `versed`'s own
  dependency tree.

- [ ] **Step 1: Write the failing test — using the REAL docstring text LangChain's own `@deprecated` decorator produces, verified directly against an installed package**

```python
# tests/unit/test_introspect_worker.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from introspect_worker import parse_deprecation  # noqa: E402

# Verbatim docstring observed from langchain==0.2.16's RetrievalQA.__doc__
REAL_RETRIEVALQA_DOC = (
    ".. deprecated:: 0.1.17 This class is deprecated. Use the "
    "`create_retrieval_chain` constructor instead. See migration guide "
    "here: https://python.langchain.com/v0.2/docs/versions/migrating_chains/retrieval_qa/\n\n"
    "Chain for question-answering against an index."
)


def test_parses_real_langchain_deprecation_notice():
    since, alternative = parse_deprecation(REAL_RETRIEVALQA_DOC)
    assert since == "0.1.17"
    assert alternative == "create_retrieval_chain"


def test_parses_auto_generated_double_backtick_form():
    doc = ".. deprecated:: 0.3.0 Use ``create_agent`` instead.\n\nSome function."
    since, alternative = parse_deprecation(doc)
    assert since == "0.3.0"
    assert alternative == "create_agent"


def test_parses_package_prefixed_since():
    doc = ".. deprecated:: langchain-core==0.3.0 Deprecated.\n\nDoc body."
    since, _ = parse_deprecation(doc)
    assert since == "0.3.0"


def test_returns_none_for_non_deprecated_docstring():
    since, alternative = parse_deprecation("A normal docstring with no notice.")
    assert since is None
    assert alternative is None


def test_returns_none_for_empty_docstring():
    assert parse_deprecation(None) == (None, None)
    assert parse_deprecation("") == (None, None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_introspect_worker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'introspect_worker'`

- [ ] **Step 3: Write `scripts/introspect_worker.py`**

```python
#!/usr/bin/env python3
"""Standalone symbol introspector.

STDLIB ONLY — no third-party imports. This script runs inside an isolated
per-version venv that has ONLY the target library installed, never this
project's own dependencies.

Usage: python introspect_worker.py <package_name>
Emits one JSON object per line (JSONL) to stdout.
"""
import importlib
import inspect
import json
import pkgutil
import re
import sys

_DEPRECATED_RE = re.compile(r"^\.\.\s*deprecated::\s*(\S+)(?:\s+(.*))?", re.DOTALL)
_ALTERNATIVE_RE = re.compile(r"[Uu]se(?: the)?\s+`{1,2}([^`]+)`{1,2}")


def parse_deprecation(doc):
    """Return (deprecated_since, alternative) parsed from a LangChain-style
    ``.. deprecated:: <since> <message>`` docstring header, or (None, None)
    if the docstring carries no deprecation notice.
    """
    if not doc:
        return None, None
    match = _DEPRECATED_RE.match(doc.strip())
    if not match:
        return None, None
    since_str = match.group(1)
    since = since_str.split("==")[-1] if "==" in since_str else since_str
    details = (match.group(2) or "").split("\n\n")[0].strip()
    alt_match = _ALTERNATIVE_RE.search(details)
    alternative = alt_match.group(1) if alt_match else None
    return since, alternative


def _safe_signature(obj):
    try:
        return str(inspect.signature(obj))
    except (ValueError, TypeError):
        return ""


def _emit(qualified_name, kind, obj):
    doc = inspect.getdoc(obj)
    since, alternative = parse_deprecation(doc)
    print(
        json.dumps(
            {
                "qualified_name": qualified_name,
                "kind": kind,
                "signature": _safe_signature(obj),
                "docstring": (doc or "")[:2000],
                "deprecated_since": since,
                "alternative": alternative,
            }
        )
    )


def walk_public_symbols(package_name):
    package = importlib.import_module(package_name)
    seen_modules = set()

    def visit_module(module):
        if module.__name__ in seen_modules:
            return
        seen_modules.add(module.__name__)

        for member_name, member in inspect.getmembers(module):
            if member_name.startswith("_"):
                continue
            if inspect.isfunction(member) and getattr(member, "__module__", None) == module.__name__:
                _emit(f"{module.__name__}.{member_name}", "function", member)
            elif inspect.isclass(member) and getattr(member, "__module__", None) == module.__name__:
                _emit(f"{module.__name__}.{member_name}", "class", member)
                for method_name, method in inspect.getmembers(member, predicate=inspect.isfunction):
                    if method_name.startswith("_"):
                        continue
                    _emit(f"{module.__name__}.{member_name}.{method_name}", "method", method)

    if hasattr(package, "__path__"):
        prefix = package.__name__ + "."
        for _, name, _ in pkgutil.walk_packages(package.__path__, prefix=prefix):
            try:
                module = importlib.import_module(name)
            except Exception:
                continue
            visit_module(module)
    visit_module(package)


if __name__ == "__main__":
    walk_public_symbols(sys.argv[1])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_introspect_worker.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Smoke-test the CLI entry point against whatever's on PATH (optional sanity check, not part of the suite)**

```bash
pip install --quiet langchain==0.2.16 2>&1 | tail -1
python3 scripts/introspect_worker.py langchain | head -3
```

Expected: several JSON lines, each with the five documented keys.

- [ ] **Step 6: Commit**

```bash
git add scripts/introspect_worker.py tests/unit/test_introspect_worker.py
git commit -m "feat: add stdlib-only symbol introspection worker"
```

---

### Task 9: Per-version venv orchestration and symbol persistence

**Files:**
- Create: `src/versed/ingest/symbol_pipeline.py`
- Test: `tests/integration/test_symbol_pipeline.py`

**Interfaces:**
- Consumes: `VersionSource` (Task 4), `Symbol` model (Task 3),
  `scripts/introspect_worker.py` (Task 8, invoked via subprocess).
- Produces: `build_symbols_for_version(source: VersionSource, venv_root: Path, session: Session) -> int`
  (returns symbol count). Task 12 (CLI) consumes this.

- [ ] **Step 1: Write `src/versed/ingest/symbol_pipeline.py`**

```python
import json
import subprocess
from pathlib import Path

from sqlalchemy.orm import Session

from versed.db.models import Symbol
from versed.ingest.manifest import VersionSource

WORKER_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "introspect_worker.py"
INTROSPECT_PYTHON = "3.11"


def create_isolated_python(venv_dir: Path) -> Path:
    subprocess.run(
        ["uv", "venv", str(venv_dir), "--python", INTROSPECT_PYTHON],
        check=True,
        capture_output=True,
        text=True,
    )
    return venv_dir / "bin" / "python"


def install_into(python_path: Path, pip_spec: str) -> None:
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python_path), pip_spec],
        check=True,
        capture_output=True,
        text=True,
    )


def run_introspection(python_path: Path, package_name: str) -> list[dict]:
    result = subprocess.run(
        [str(python_path), str(WORKER_SCRIPT), package_name],
        check=True,
        capture_output=True,
        text=True,
    )
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def build_symbols_for_version(source: VersionSource, venv_root: Path, session: Session) -> int:
    venv_dir = venv_root / source.version.replace(".", "_")
    python_path = create_isolated_python(venv_dir)
    install_into(python_path, source.pip_spec)

    package_name = source.pip_spec.split("==")[0]
    records = run_introspection(python_path, package_name)

    session.query(Symbol).filter_by(version=source.version).delete()
    for record in records:
        session.add(
            Symbol(
                version=source.version,
                qualified_name=record["qualified_name"],
                kind=record["kind"],
                signature=record["signature"],
                docstring=record["docstring"] or None,
                deprecated_since=record["deprecated_since"],
                alternative=record["alternative"],
            )
        )
    session.commit()
    return len(records)
```

- [ ] **Step 2: Write the integration test**

```python
# tests/integration/test_symbol_pipeline.py
import pytest
from sqlalchemy import select

from versed.db.models import Symbol
from versed.db.session import get_session
from versed.ingest.manifest import VersionSource
from versed.ingest.symbol_pipeline import build_symbols_for_version

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_build_symbols_for_version_finds_known_deprecated_class(tmp_path):
    source = VersionSource(
        version="0.2",
        repo_url="https://github.com/langchain-ai/langchain.git",
        ref="langchain==0.2.0",
        docs_subpath="docs/docs",
        pip_spec="langchain==0.2.0",
    )
    with get_session() as session:
        session.query(Symbol).filter_by(version="0.2").delete()
        session.commit()

        count = build_symbols_for_version(source, tmp_path / "venvs", session)
        assert count > 0

        retrieval_qa = session.scalars(
            select(Symbol).where(
                Symbol.version == "0.2",
                Symbol.qualified_name.like("%.RetrievalQA"),
            )
        ).first()
        assert retrieval_qa is not None
        assert retrieval_qa.deprecated_since is not None
```

- [ ] **Step 3: Run it (slow — creates a venv and pip-installs a real package)**

Run: `uv run pytest tests/integration/test_symbol_pipeline.py -v -m integration`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/versed/ingest/symbol_pipeline.py tests/integration/test_symbol_pipeline.py
git commit -m "feat: add per-version venv orchestration and symbol persistence"
```

---

### Task 10: Symbol diff logic

**Files:**
- Create: `src/versed/ingest/diff.py`
- Test: `tests/unit/test_diff.py`

**Interfaces:**
- Produces: `SymbolRow` dataclass (`qualified_name`, `kind`, `signature`,
  `deprecated_since`, `alternative`), `DiffEvent` dataclass
  (`qualified_name`, `from_version`, `to_version`, `event_type`, `detail`),
  `diff_versions(from_version, from_symbols, to_version, to_symbols) -> list[DiffEvent]`.
  Task 11 consumes all three.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_diff.py
from versed.ingest.diff import SymbolRow, diff_versions


def row(name, sig="()", deprecated_since=None, alternative=None):
    return SymbolRow(
        qualified_name=name,
        kind="function",
        signature=sig,
        deprecated_since=deprecated_since,
        alternative=alternative,
    )


def test_detects_added_symbol():
    events = diff_versions("0.1", {}, "0.2", {"langchain.foo": row("langchain.foo")})
    assert len(events) == 1
    assert events[0].event_type == "added"
    assert events[0].qualified_name == "langchain.foo"


def test_detects_removed_symbol():
    events = diff_versions("0.1", {"langchain.foo": row("langchain.foo")}, "0.2", {})
    assert len(events) == 1
    assert events[0].event_type == "removed"


def test_detects_newly_deprecated_symbol():
    before = {"langchain.chains.RetrievalQA": row("langchain.chains.RetrievalQA")}
    after = {
        "langchain.chains.RetrievalQA": row(
            "langchain.chains.RetrievalQA",
            deprecated_since="0.1.17",
            alternative="create_retrieval_chain",
        )
    }
    events = diff_versions("0.1", before, "0.2", after)
    assert len(events) == 1
    assert events[0].event_type == "deprecated"
    assert events[0].detail == "create_retrieval_chain"


def test_detects_changed_signature():
    before = {"langchain.agents.create_agent": row("langchain.agents.create_agent", sig="(prompt)")}
    after = {"langchain.agents.create_agent": row("langchain.agents.create_agent", sig="(system_prompt)")}
    events = diff_versions("0.3", before, "1.0", after)
    assert len(events) == 1
    assert events[0].event_type == "changed"
    assert events[0].detail == "(prompt) -> (system_prompt)"


def test_pairs_rename_into_single_moved_event():
    old_name = "langchain.retrievers.ContextualCompressionRetriever"
    new_name = "langchain_classic.retrievers.contextual_compression.ContextualCompressionRetriever"
    before = {old_name: row(old_name)}
    after = {new_name: row(new_name)}
    events = diff_versions("0.3", before, "1.0", after)
    assert len(events) == 1
    assert events[0].event_type == "moved"
    assert events[0].qualified_name == new_name
    assert old_name in events[0].detail
    assert new_name in events[0].detail


def test_unchanged_symbol_produces_no_event():
    before = {"langchain.x": row("langchain.x", sig="()")}
    after = {"langchain.x": row("langchain.x", sig="()")}
    assert diff_versions("0.1", before, "0.2", after) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_diff.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `src/versed/ingest/diff.py`**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolRow:
    qualified_name: str
    kind: str
    signature: str
    deprecated_since: str | None
    alternative: str | None


@dataclass(frozen=True)
class DiffEvent:
    qualified_name: str
    from_version: str | None
    to_version: str
    event_type: str
    detail: str | None


def _short_name(qualified_name: str) -> str:
    return qualified_name.rsplit(".", 1)[-1]


def diff_versions(
    from_version: str,
    from_symbols: dict[str, SymbolRow],
    to_version: str,
    to_symbols: dict[str, SymbolRow],
) -> list[DiffEvent]:
    events: list[DiffEvent] = []

    removed_names = set(from_symbols) - set(to_symbols)
    added_names = set(to_symbols) - set(from_symbols)
    common_names = set(from_symbols) & set(to_symbols)

    removed_by_short: dict[str, list[str]] = {}
    for name in removed_names:
        removed_by_short.setdefault(_short_name(name), []).append(name)

    matched_added: set[str] = set()
    matched_removed: set[str] = set()
    moved_pairs: list[tuple[str, str]] = []
    for added_name in sorted(added_names):
        candidates = removed_by_short.get(_short_name(added_name), [])
        if len(candidates) == 1:
            old_name = candidates[0]
            moved_pairs.append((old_name, added_name))
            matched_added.add(added_name)
            matched_removed.add(old_name)

    for old_name, new_name in moved_pairs:
        events.append(
            DiffEvent(
                qualified_name=new_name,
                from_version=from_version,
                to_version=to_version,
                event_type="moved",
                detail=f"Moved from {old_name} to {new_name}",
            )
        )

    for name in sorted(added_names - matched_added):
        events.append(
            DiffEvent(
                qualified_name=name,
                from_version=from_version,
                to_version=to_version,
                event_type="added",
                detail=None,
            )
        )

    for name in sorted(removed_names - matched_removed):
        events.append(
            DiffEvent(
                qualified_name=name,
                from_version=from_version,
                to_version=to_version,
                event_type="removed",
                detail=None,
            )
        )

    for name in sorted(common_names):
        before = from_symbols[name]
        after = to_symbols[name]
        if after.deprecated_since and not before.deprecated_since:
            events.append(
                DiffEvent(
                    qualified_name=name,
                    from_version=from_version,
                    to_version=to_version,
                    event_type="deprecated",
                    detail=after.alternative,
                )
            )
        elif before.signature != after.signature:
            events.append(
                DiffEvent(
                    qualified_name=name,
                    from_version=from_version,
                    to_version=to_version,
                    event_type="changed",
                    detail=f"{before.signature} -> {after.signature}",
                )
            )

    return events
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_diff.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/versed/ingest/diff.py tests/unit/test_diff.py
git commit -m "feat: add symbol diff logic (added/removed/deprecated/changed/moved)"
```

---

### Task 11: Timeline pipeline

**Files:**
- Create: `src/versed/ingest/timeline_pipeline.py`
- Test: `tests/integration/test_timeline_pipeline.py`

**Interfaces:**
- Consumes: `Symbol`/`SymbolEvent` models (Task 3), `SymbolRow`,
  `diff_versions` (Task 10).
- Produces: `build_timeline(ordered_versions: list[str], session: Session) -> int`
  (returns total event count). Task 12 (CLI) consumes this.

- [ ] **Step 1: Write `src/versed/ingest/timeline_pipeline.py`**

```python
from sqlalchemy import select
from sqlalchemy.orm import Session

from versed.db.models import Symbol, SymbolEvent
from versed.ingest.diff import SymbolRow, diff_versions


def _load_symbols(version: str, session: Session) -> dict[str, SymbolRow]:
    rows = session.scalars(select(Symbol).where(Symbol.version == version)).all()
    return {
        row.qualified_name: SymbolRow(
            qualified_name=row.qualified_name,
            kind=row.kind,
            signature=row.signature,
            deprecated_since=row.deprecated_since,
            alternative=row.alternative,
        )
        for row in rows
    }


def build_timeline(ordered_versions: list[str], session: Session) -> int:
    session.query(SymbolEvent).delete()
    total = 0
    for from_version, to_version in zip(ordered_versions, ordered_versions[1:]):
        from_symbols = _load_symbols(from_version, session)
        to_symbols = _load_symbols(to_version, session)
        events = diff_versions(from_version, from_symbols, to_version, to_symbols)
        for event in events:
            session.add(
                SymbolEvent(
                    qualified_name=event.qualified_name,
                    from_version=event.from_version,
                    to_version=event.to_version,
                    event_type=event.event_type,
                    detail=event.detail,
                )
            )
        total += len(events)
    session.commit()
    return total
```

- [ ] **Step 2: Write the integration test**

```python
# tests/integration/test_timeline_pipeline.py
import pytest
from sqlalchemy import select

from versed.db.models import Symbol, SymbolEvent
from versed.db.session import get_session
from versed.ingest.timeline_pipeline import build_timeline

pytestmark = pytest.mark.integration


def test_build_timeline_across_three_synthetic_versions():
    with get_session() as session:
        session.query(SymbolEvent).delete()
        session.query(Symbol).filter(Symbol.version.in_(["t1", "t2", "t3"])).delete()

        session.add(Symbol(version="t1", qualified_name="pkg.old_fn", kind="function", signature="()"))
        session.add(
            Symbol(
                version="t2",
                qualified_name="pkg.old_fn",
                kind="function",
                signature="()",
                deprecated_since="t2",
                alternative="pkg.new_fn",
            )
        )
        session.add(Symbol(version="t3", qualified_name="pkg.new_fn", kind="function", signature="()"))
        session.commit()

        total = build_timeline(["t1", "t2", "t3"], session)
        # t1->t2: 1 "deprecated" event (pkg.old_fn gains deprecated_since).
        # t2->t3: 2 events, NOT 1 "moved" — "old_fn" and "new_fn" have
        # different short names, so Task 10's exact-short-name move-pairing
        # correctly does not merge them; it's "removed" pkg.old_fn +
        # "added" pkg.new_fn. Total: 3.
        assert total == 3

        events = session.scalars(
            select(SymbolEvent).where(SymbolEvent.qualified_name.like("%old_fn%"))
        ).all()
        event_types = {e.event_type for e in events}
        assert "deprecated" in event_types
```

- [ ] **Step 3: Run it**

Run: `uv run pytest tests/integration/test_timeline_pipeline.py -v -m integration`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/versed/ingest/timeline_pipeline.py tests/integration/test_timeline_pipeline.py
git commit -m "feat: add timeline pipeline (diffs consecutive versions into events)"
```

---

### Task 12: CLI — ingest, build-symbols, build-timeline, timeline query

**Files:**
- Create: `src/versed/ingest/symbol_query.py`
- Create: `src/versed/cli.py`
- Test: `tests/integration/test_symbol_query.py`

**Interfaces:**
- Consumes: everything from Tasks 3–11.
- Produces: `symbol_history(qualified_name_suffix: str, session: Session) -> list[SymbolEvent]`
  from `versed.ingest.symbol_query`; a Typer `app` in `versed.cli` with
  commands `ingest-docs`, `build-symbols`, `build-timeline`, `timeline
  <symbol>` — this task's and this plan's demo deliverable.

- [ ] **Step 1: Write `src/versed/ingest/symbol_query.py`**

```python
from sqlalchemy import select
from sqlalchemy.orm import Session

from versed.db.models import SymbolEvent


def symbol_history(qualified_name_suffix: str, session: Session) -> list[SymbolEvent]:
    """Return every event whose qualified_name ends with the given suffix,
    ordered chronologically. Suffix match lets callers query by short name
    (e.g. "create_agent") without knowing the full module path.
    """
    stmt = (
        select(SymbolEvent)
        .where(SymbolEvent.qualified_name.like(f"%{qualified_name_suffix}"))
        .order_by(SymbolEvent.to_version)
    )
    return list(session.scalars(stmt).all())
```

- [ ] **Step 2: Write the integration test**

```python
# tests/integration/test_symbol_query.py
import pytest

from versed.db.models import SymbolEvent
from versed.db.session import get_session
from versed.ingest.symbol_query import symbol_history

pytestmark = pytest.mark.integration


def test_symbol_history_matches_by_suffix_and_orders_chronologically():
    with get_session() as session:
        session.query(SymbolEvent).filter(
            SymbolEvent.qualified_name.like("%probe_symbol_query%")
        ).delete()
        session.add(
            SymbolEvent(
                qualified_name="pkg.b.probe_symbol_query",
                from_version="0.2",
                to_version="0.3",
                event_type="added",
                detail=None,
            )
        )
        session.add(
            SymbolEvent(
                qualified_name="pkg.b.probe_symbol_query",
                from_version="0.3",
                to_version="1.0",
                event_type="deprecated",
                detail="pkg.c.new_probe",
            )
        )
        session.commit()

        events = symbol_history("probe_symbol_query", session)
        assert [e.to_version for e in events] == ["0.3", "1.0"]
        assert [e.event_type for e in events] == ["added", "deprecated"]
```

- [ ] **Step 3: Run it**

Run: `uv run pytest tests/integration/test_symbol_query.py -v -m integration`
Expected: PASS

- [ ] **Step 4: Write `src/versed/cli.py`**

```python
from pathlib import Path

import typer

from versed.db.session import get_session
from versed.ingest.docs_pipeline import ingest_docs_for_version
from versed.ingest.manifest import build_manifest
from versed.ingest.symbol_pipeline import build_symbols_for_version
from versed.ingest.symbol_query import symbol_history
from versed.ingest.timeline_pipeline import build_timeline

app = typer.Typer()


@app.command("ingest-docs")
def ingest_docs_cmd() -> None:
    manifest = build_manifest()
    with get_session() as session:
        for source in manifest:
            count = ingest_docs_for_version(source, session)
            typer.echo(f"{source.version}: {count} chunks")


@app.command("build-symbols")
def build_symbols_cmd(venv_root: Path = Path(".versed-venvs")) -> None:
    manifest = build_manifest()
    venv_root.mkdir(exist_ok=True)
    with get_session() as session:
        for source in manifest:
            count = build_symbols_for_version(source, venv_root, session)
            typer.echo(f"{source.version}: {count} symbols")


@app.command("build-timeline")
def build_timeline_cmd() -> None:
    manifest = build_manifest()
    versions = [source.version for source in manifest]
    with get_session() as session:
        count = build_timeline(versions, session)
        typer.echo(f"{count} events across {len(versions) - 1} version transitions")


@app.command("timeline")
def timeline_cmd(symbol: str) -> None:
    with get_session() as session:
        events = symbol_history(symbol, session)
        if not events:
            typer.echo(f"No history found for '{symbol}'")
            raise typer.Exit(code=1)
        for event in events:
            arrow = (
                f"{event.from_version} -> {event.to_version}"
                if event.from_version
                else event.to_version
            )
            detail = f" ({event.detail})" if event.detail else ""
            typer.echo(f"[{arrow}] {event.event_type}: {event.qualified_name}{detail}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Run the full pipeline end to end and produce the milestone deliverable**

```bash
docker compose up -d db
uv run alembic upgrade head
uv run python -m versed.cli ingest-docs
uv run python -m versed.cli build-symbols
uv run python -m versed.cli build-timeline
uv run python -m versed.cli timeline create_agent
```

Expected: the last command prints at least one `moved` or `added` line for
`create_agent` (exact events depend on what the real 0.1–current diff
surfaces — record the actual output in the README, don't assume the number
in advance).

- [ ] **Step 6: Commit**

```bash
git add src/versed/ingest/symbol_query.py src/versed/cli.py tests/integration/test_symbol_query.py
git commit -m "feat: add CLI (ingest-docs, build-symbols, build-timeline, timeline query)"
```

---

### Task 12B: Introspect langchain_classic for accurate move detection

**Files:**
- Modify: `src/versed/ingest/manifest.py`
- Modify: `src/versed/ingest/symbol_pipeline.py`
- Modify: `tests/unit/test_manifest.py`
- Modify: `tests/integration/test_symbol_pipeline.py`

**Interfaces:**
- Produces: `VersionSource` gains an `extra_pip_specs: list[str]` field
  (default empty). `build_symbols_for_version` installs and introspects
  each entry in `extra_pip_specs` into the SAME isolated venv as the main
  `pip_spec`, persisting all resulting symbols under the same `version`.
- Reason for this task: the real pipeline run (Task 12) surfaced that
  `langchain`'s current version has only ~340 symbols vs ~7500 for v0.3 —
  verified real, not a bug (LangChain 1.0 moved most legacy code to a
  separate `langchain_classic` package). Since only `langchain` itself is
  introspected, a symbol that moved to `langchain_classic` currently shows
  as a bare `removed` event — true, but not as useful as `moved`, which
  Task 10's existing move-pairing logic would produce automatically if it
  could see the symbol on both sides. No changes to `diff.py` are needed —
  only to what gets introspected.
- **Verified before writing this task, not assumed:** `langchain-classic`
  does NOT share `langchain`'s version number (checked live via PyPI:
  `langchain-classic` was at `1.0.8` while `langchain` was at `1.4.0`), so
  it must be resolved independently, not reused from the already-resolved
  `latest` value.

- [ ] **Step 1: Add a reusable PyPI-version resolver to `manifest.py`, and the new field**

Modify `src/versed/ingest/manifest.py`:

```python
from dataclasses import dataclass, field

import httpx


@dataclass(frozen=True)
class VersionSource:
    version: str
    repo_url: str
    ref: str
    docs_subpath: str
    pip_spec: str
    extra_pip_specs: list[str] = field(default_factory=list)


def resolve_latest_pypi_version(package: str) -> str:
    response = httpx.get(f"https://pypi.org/pypi/{package}/json", timeout=10.0)
    response.raise_for_status()
    return response.json()["info"]["version"]


def resolve_latest_langchain_version() -> str:
    return resolve_latest_pypi_version("langchain")
```

`resolve_latest_langchain_version()` keeps its exact existing signature and
behavior (still no-arg, still resolves `langchain` specifically) so nothing
that already calls it needs to change — it's now a one-line wrapper around
the new, reusable, package-parameterized function.

- [ ] **Step 2: Populate `extra_pip_specs` for the current version only**

In `build_manifest()`, add one line resolving `langchain-classic`'s own
latest version, and add `extra_pip_specs` to the `latest` `VersionSource`
entry only — 0.1/0.2/0.3 get no extra packages, since `langchain_classic`
did not exist as a separate package before LangChain 1.0:

```python
def build_manifest() -> list[VersionSource]:
    latest = resolve_latest_langchain_version()
    latest_classic = resolve_latest_pypi_version("langchain-classic")
    langchain_repo = "https://github.com/langchain-ai/langchain.git"
    return [
        VersionSource(
            version="0.1",
            repo_url=langchain_repo,
            ref="v0.1.0",
            docs_subpath="docs/docs",
            pip_spec="langchain==0.1.0",
        ),
        VersionSource(
            version="0.2",
            repo_url=langchain_repo,
            ref="langchain==0.2.0",
            docs_subpath="docs/docs",
            pip_spec="langchain==0.2.0",
        ),
        VersionSource(
            version="0.3",
            repo_url=langchain_repo,
            ref="langchain==0.3.0",
            docs_subpath="docs/docs",
            pip_spec="langchain==0.3.0",
        ),
        VersionSource(
            version=latest,
            repo_url="https://github.com/langchain-ai/docs.git",
            ref="main",
            docs_subpath="src/oss/python",
            pip_spec=f"langchain=={latest}",
            extra_pip_specs=[f"langchain-classic=={latest_classic}"],
        ),
    ]
```

- [ ] **Step 3: Update `tests/unit/test_manifest.py` with a URL-aware fake**

The existing fake response returns the same version regardless of which
package was requested — too weak to prove `langchain-classic` is resolved
independently. Replace it with one that branches on the request URL:

```python
# tests/unit/test_manifest.py
import httpx

from versed.ingest.manifest import build_manifest, resolve_latest_langchain_version


class _FakeResponse:
    def __init__(self, version: str):
        self._version = version

    def raise_for_status(self):
        pass

    def json(self):
        return {"info": {"version": self._version}}


def _fake_get(url, *a, **k):
    if "langchain-classic" in url:
        return _FakeResponse("1.0.8")
    return _FakeResponse("1.9.9")


def test_resolve_latest_langchain_version(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get)
    assert resolve_latest_langchain_version() == "1.9.9"


def test_build_manifest_has_four_versions(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get)
    manifest = build_manifest()
    assert [s.version for s in manifest] == ["0.1", "0.2", "0.3", "1.9.9"]
    assert manifest[0].ref == "v0.1.0"
    assert manifest[2].ref == "langchain==0.3.0"
    assert manifest[3].repo_url == "https://github.com/langchain-ai/docs.git"
    assert manifest[3].pip_spec == "langchain==1.9.9"


def test_only_current_version_has_extra_pip_specs(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get)
    manifest = build_manifest()
    assert manifest[0].extra_pip_specs == []
    assert manifest[1].extra_pip_specs == []
    assert manifest[2].extra_pip_specs == []
    assert manifest[3].extra_pip_specs == ["langchain-classic==1.0.8"]
```

This replaces the two existing test functions with the same names (updated
to use `_fake_get`) plus one new test — a net one-function addition.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_manifest.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Extend `symbol_pipeline.py` to install and introspect extra packages**

Modify `src/versed/ingest/symbol_pipeline.py`:

```python
def _pip_spec_to_import_name(pip_spec: str) -> str:
    return pip_spec.split("==")[0].replace("-", "_")


def build_symbols_for_version(source: VersionSource, venv_root: Path, session: Session) -> int:
    venv_dir = venv_root / source.version.replace(".", "_")
    python_path = create_isolated_python(venv_dir)
    install_into(python_path, source.pip_spec)

    records = run_introspection(python_path, _pip_spec_to_import_name(source.pip_spec))

    for extra_spec in source.extra_pip_specs:
        install_into(python_path, extra_spec)
        records.extend(run_introspection(python_path, _pip_spec_to_import_name(extra_spec)))

    session.query(Symbol).filter_by(version=source.version).delete()
    for record in records:
        session.add(
            Symbol(
                version=source.version,
                qualified_name=record["qualified_name"],
                kind=record["kind"],
                signature=record["signature"],
                docstring=record["docstring"] or None,
                deprecated_since=record["deprecated_since"],
                alternative=record["alternative"],
            )
        )
    session.commit()
    return len(records)
```

`_pip_spec_to_import_name` replaces the old inline
`source.pip_spec.split("==")[0]` — same result for `"langchain"` (no
hyphen, no behavior change), but now also correct for hyphenated pip names
like `langchain-classic` (pip name) vs `langchain_classic` (import name),
which the old inline version would have gotten wrong.

- [ ] **Step 6: Add an integration test proving both packages get introspected**

```python
# tests/integration/test_symbol_pipeline.py — add this test to the existing file
def test_build_symbols_for_version_installs_extra_packages(tmp_path):
    source = VersionSource(
        version="1.0.8-classic-test",
        repo_url="https://github.com/langchain-ai/langchain.git",
        ref="langchain==1.0.0",
        docs_subpath="docs/docs",
        pip_spec="langchain==1.0.0",
        extra_pip_specs=["langchain-classic==1.0.8"],
    )
    with get_session() as session:
        session.query(Symbol).filter_by(version="1.0.8-classic-test").delete()
        session.commit()

        count = build_symbols_for_version(source, tmp_path / "venvs", session)
        assert count > 0

        langchain_symbols = session.scalars(
            select(Symbol).where(
                Symbol.version == "1.0.8-classic-test",
                Symbol.qualified_name.like("langchain.%"),
            )
        ).all()
        classic_symbols = session.scalars(
            select(Symbol).where(
                Symbol.version == "1.0.8-classic-test",
                Symbol.qualified_name.like("langchain_classic.%"),
            )
        ).all()
        assert len(langchain_symbols) > 0
        assert len(classic_symbols) > 0
```

(Add the `get_session` import if not already present in this test file —
it already imports it for the existing test in this file.)

- [ ] **Step 7: Run it**

Run: `uv run pytest tests/integration/test_symbol_pipeline.py -v -m integration`
Expected: PASS (both tests — the existing one and the new one)

- [ ] **Step 8: Verify the actual fix live — re-run the real pipeline for the current version and confirm a real relocation now resolves to `moved`**

```bash
uv run python -m versed.cli build-symbols
uv run python -m versed.cli build-timeline
uv run python -m versed.cli timeline ContextualCompressionRetriever
```

Expected: a `moved` line appears for `ContextualCompressionRetriever`
(confirmed in Task 10's review to genuinely exist at both
`langchain.retrievers.ContextualCompressionRetriever` in 0.3 and
`langchain_classic.retrievers.contextual_compression.ContextualCompressionRetriever`
in current) — where before this task it would have shown as a bare
`removed` event with no counterpart. Record the actual output; don't
assume it matches this description exactly without checking.

- [ ] **Step 9: Update the Global Constraints note this task addresses**

In the plan's Global Constraints, the "Known scope limit" bullet currently
says relocated symbols "will not appear as a `moved` event." Add a short
note that this is now resolved for `langchain` → `langchain_classic`
relocations specifically (this task), while the separate `langgraph` cross
-package case remains open (a different package, not addressed here).

- [ ] **Step 10: Commit**

```bash
git add src/versed/ingest/manifest.py src/versed/ingest/symbol_pipeline.py tests/unit/test_manifest.py tests/integration/test_symbol_pipeline.py docs/superpowers/plans/2026-09-03-versed-foundation.md
git commit -m "feat: introspect langchain_classic alongside langchain for accurate move detection"
```

---

### Task 12C: Fix venv recreation for re-runnable build-symbols

**Files:**
- Modify: `src/versed/ingest/symbol_pipeline.py`
- Modify: `tests/integration/test_symbol_pipeline.py`

**Interfaces:**
- No signature changes. `create_isolated_python(venv_dir: Path) -> Path`
  keeps its exact contract; it just stops failing when `venv_dir` already
  has content in it.
- Reason: Task 12B's implementer found that a second `build-symbols` run
  against a non-empty `.versed-venvs/` cache hard-fails, because `uv venv`
  refuses to write into an existing directory by default. This matters for
  the project's incremental-rerun design — `build_manifest()` already
  re-resolves `latest`/`latest_classic` live from PyPI specifically so
  `build-symbols` can be re-run as new releases ship, without code
  changes; a venv-creation failure defeats that unless the cache is
  manually cleared first every time.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_symbol_pipeline.py — add to the existing file
def test_create_isolated_python_succeeds_on_existing_venv_dir(tmp_path):
    venv_dir = tmp_path / "reused_venv"

    first_python = create_isolated_python(venv_dir)
    assert first_python.exists()

    # Re-running against the same, now-populated directory must not raise.
    second_python = create_isolated_python(venv_dir)
    assert second_python.exists()
```

(Add `create_isolated_python` to this file's existing import from
`versed.ingest.symbol_pipeline` if not already imported.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_symbol_pipeline.py::test_create_isolated_python_succeeds_on_existing_venv_dir -v -m integration`
Expected: FAIL — `subprocess.CalledProcessError` on the second
`create_isolated_python` call (`uv venv` refuses to write into the
already-populated directory).

- [ ] **Step 3: Add `--clear` to the `uv venv` call**

In `src/versed/ingest/symbol_pipeline.py`:

```python
def create_isolated_python(venv_dir: Path) -> Path:
    subprocess.run(
        ["uv", "venv", str(venv_dir), "--python", INTROSPECT_PYTHON, "--clear"],
        check=True,
        capture_output=True,
        text=True,
    )
    return venv_dir / "bin" / "python"
```

`--clear` is `uv venv`'s own native flag for exactly this ("Remove any
existing files or directories at the target path") — not a hand-rolled
`rm -rf` before the call. Verified this flag is real and does what's
described via `uv venv --help` before writing this task, not assumed from
memory.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_symbol_pipeline.py::test_create_isolated_python_succeeds_on_existing_venv_dir -v -m integration`
Expected: PASS

- [ ] **Step 5: Run the full existing test file to confirm nothing else broke**

Run: `uv run pytest tests/integration/test_symbol_pipeline.py -v -m integration`
Expected: PASS (all tests, including the pre-existing ones from Tasks 9 and 12B)

- [ ] **Step 6: Commit**

```bash
git add src/versed/ingest/symbol_pipeline.py tests/integration/test_symbol_pipeline.py
git commit -m "fix: allow build-symbols to re-run against an existing venv cache"
```

---

### Task 13: CI — lint, type-check, unit tests

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `pyproject.toml` (add `mypy` to the dev dependency group, add
  `[tool.mypy]` config)

**Interfaces:**
- Produces: a GitHub Actions workflow with three jobs (lint, typecheck,
  test) running on every push to `main` and every PR — not integration/slow
  tests, which need Postgres, a running Ollama daemon with `nomic-embed-text`
  pulled, and real network installs, none of which a hosted GitHub Actions
  runner has by default, and stay out of scope for this plan's CI (the full
  eval-gating CI belongs to the Evaluation plan). Separate jobs, not one
  script, so a failure names its category in the PR checks list rather than
  requiring a log read.
- Uses libraries instead of hand-rolled checks wherever one exists: `ruff`
  for both lint and format (one tool, two modes, instead of separate
  lint/format tools), `mypy` for types, `astral-sh/setup-uv` (official
  action) for toolchain setup — no custom install scripts.

- [ ] **Step 1: Add mypy and its config to `pyproject.toml`**

Add to the `dev` dependency group:

```toml
    "mypy>=1.11",
```

Add a new top-level table:

```toml
[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true
warn_unused_ignores = true
warn_redundant_casts = true
```

`ignore_missing_imports = true` is deliberate, not a placeholder relaxation:
several dependencies in this project (e.g. early-stage libraries without
published type stubs) may lack types, and failing CI on a third-party
library's missing stubs is a false positive this project doesn't own. Types
in `src/versed` itself are still fully checked.

- [ ] **Step 2: Run mypy locally and fix any real findings**

```bash
uv sync --all-groups
uv run mypy src/versed
```

Expected: exits 0. If it doesn't, fix the flagged `src/versed` code — do not
loosen `[tool.mypy]` further to silence a real type error.

- [ ] **Step 3: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --all-groups
      - name: Ruff check
        run: uv run ruff check .
      - name: Ruff format check
        run: uv run ruff format --check .

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --all-groups
      - name: mypy
        run: uv run mypy src/versed

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --all-groups
      - name: Unit tests
        run: uv run pytest tests/unit -v

  dependency-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - name: pip-audit (dependency vulnerability scan)
        run: uvx pip-audit --strict
```

`dependency-audit` runs `pip-audit` via `uvx` (uv's ephemeral tool runner) —
no dependency added to the project itself for a tool only CI needs.
`--strict` fails the job on any known vulnerability rather than only
warning, matching this plan's general stance that a check which never fails
isn't a gate.

- [ ] **Step 4: Verify every job locally before pushing**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/versed
uv run pytest tests/unit -v
uvx pip-audit --strict
```

Expected: all five succeed. If `ruff format --check` fails, run
`uv run ruff format .` once to fix formatting, then re-verify.

- [ ] **Step 5: Commit and push**

```bash
git add pyproject.toml .github/workflows/ci.yml
git commit -m "ci: add lint, format check, type check, unit tests, and dependency audit"
```

---

### Task 14: Security scanning and dependency updates

**Files:**
- Create: `.github/workflows/codeql.yml`
- Create: `.github/dependabot.yml`

**Interfaces:**
- Produces: automated SAST (CodeQL) on every push/PR and weekly on a
  schedule, plus automated dependency-update PRs (uv-managed Python deps and
  GitHub Actions versions). Both are GitHub-native services configured via
  files, not custom code — the whole point of this task is using existing
  infrastructure instead of writing a vulnerability scanner or an update bot.
- This task is independent of Task 13 (no shared files) — it can be done in
  either order relative to it.

- [ ] **Step 1: Write `.github/workflows/codeql.yml`**

```yaml
name: CodeQL

on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: "17 3 * * 1"

jobs:
  analyze:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      contents: read
    strategy:
      matrix:
        language: ["python"]
    steps:
      - uses: actions/checkout@v4
      - uses: github/codeql-action/init@v4
        with:
          languages: ${{ matrix.language }}
      - uses: github/codeql-action/analyze@v4
```

Pinned to v4, not v3: GitHub's own runner output on this task's first real
run flagged that CodeQL Action v3 is deprecated in December 2026 — since
this is a brand-new workflow with no legacy v3 usage to migrate, there's
no reason to start on the version already scheduled for deprecation.
Verified `v4` is a real, published tag (not just announced) before
switching, via the GitHub API, not assumed from the deprecation notice
alone.

The weekly cron (Monday 03:17 UTC — an off-peak, non-round time to avoid
GitHub's documented top-of-hour scheduling congestion) catches vulnerable
patterns in code that hasn't changed recently but where CodeQL's own rule
set has been updated since the last push.

- [ ] **Step 2: Write `.github/dependabot.yml`**

```yaml
version: 2
updates:
  - package-ecosystem: "uv"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

The `uv` ecosystem (not `pip`) is used deliberately — Dependabot's native uv
support understands `uv.lock` directly. Known rough edge as of this writing:
some reports of Dependabot updating `uv.lock` without updating the matching
`pyproject.toml` line, or vice versa. When the first Dependabot PR lands,
verify both files changed together before merging; if only one did, that's
a real Dependabot bug to work around (e.g. by re-running `uv lock` locally
on that PR's branch), not a config error in this file.

- [ ] **Step 3: Verify the YAML is well-formed**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/codeql.yml'))"
python3 -c "import yaml; yaml.safe_load(open('.github/dependabot.yml'))"
```

Expected: both exit 0 with no output (valid YAML). This does not verify
GitHub accepts the semantics — that's confirmed after push in Step 5.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/codeql.yml .github/dependabot.yml
git commit -m "ci: add CodeQL security scanning and Dependabot updates"
```

- [ ] **Step 5: Push and confirm both are live**

```bash
git push
gh workflow list
```

Expected: `CodeQL` appears in the workflow list and its first run completes
(check via `gh run list --workflow=codeql.yml`). Dependabot's first scan is
not immediate — confirm it under the repo's Insights → Dependency graph →
Dependabot tab within the following day, or via
`gh api repos/:owner/:repo/dependabot/alerts` once available.

---

## What this plan does not cover (by design)

Hybrid retrieval, the LangGraph answer graph, the golden eval set, guardrails,
FastAPI serving, and production ops are Milestones 3–8 of the spec and belong
to later plans, built on top of the `chunk`/`symbol`/`symbol_event` tables
this plan produces. Extending symbol introspection to also cover `langgraph`
(to capture the `create_react_agent` → `create_agent` cross-package move) is
a small, well-contained follow-up to `manifest.py` and `symbol_pipeline.py`
once this plan is merged — flagged here rather than silently assumed away.

Task 13/14's CI is deliberately scoped to what a project this size needs now
(lint, format, types, unit tests, dependency audit, CodeQL, Dependabot) —
not a full DevSecOps platform. Out of scope here, revisit once the service
is deployed (Milestone 8): container image scanning (e.g. Trivy) once a
Dockerfile exists, required branch-protection status checks (set once the
job names in Task 13/14 are stable and have run green at least once — a
protection rule pointed at a check that has never reported blocks every
merge), and SAST beyond CodeQL's default query set.

## Final review for this plan

Once all 14 tasks are merged to `main`, this plan's whole-branch-equivalent
review (per superpowers:subagent-driven-development's Final Review step,
adapted for this project's per-task-PR workflow to mean "review the full
set of changes since this plan started") runs two passes, not one:
`superpowers:requesting-code-review`'s code-reviewer for correctness and the
library-reuse/simplification criterion above, and the `security-review`
skill for anything CodeQL's default queries don't catch (secrets, injection
surfaces, trust boundaries specific to this project's design — e.g. the
`introspect_worker.py` subprocess boundary, the git-ref fetcher's use of
user-influenced version strings). Findings from either pass are ledgered and
fixed the same way as a task-level finding.

The final review ran on 2026-09-06 (code-reviewer, security-review, and
ponytail-audit/ponytail-debt, all on the full 0648d90..83751d8 range).
Findings are grouped into follow-up tasks by theme rather than one task per
finding, and executed in parallel using isolated git worktrees per group
(mechanical fixes with no shared state between groups, verified before
dispatch that any file touched by two groups only overlaps in
non-conflicting regions).

---

### Task 15: Final-review fixes — security

Addresses: pip-audit scanning the wrong packages (confirmed independently
by both the code-reviewer and security-review passes — the strongest-signal
finding of the whole review); Postgres bound to `0.0.0.0` instead of
loopback when its committed dev credentials are the only real barrier;
missing `--` before positional args in the two subprocess calls that take a
value ultimately sourced from an external system (git ref/repo, pip spec) —
not exploitable today since `manifest.py`'s values are hardcoded literals,
but the cheap, permanent guard rail against the day that changes; a missing
`permissions:` block on `ci.yml`; and documenting (not redesigning) the
introspection worker's real trust boundary, since its own docstring
currently implies more isolation than a venv actually provides.

- pip-audit: export the real lock file first (`uv export --frozen
  --all-groups --format requirements-txt --no-emit-project -o
  requirements.txt`), then `uvx pip-audit --strict -r requirements.txt` —
  verified locally before writing this: the old command audited 28
  packages (its own ephemeral `uvx` environment), the fixed command audits
  the real 73.
- `docker-compose.yml`: `"5433:5432"` → `"127.0.0.1:5433:5432"`.
- `src/versed/ingest/docs_fetch.py`: add `--` before `repo_url` in the
  `git clone` argv list. Verified working via a real local clone before
  writing this task (`git clone --depth 1 --branch <ref> -- <repo> <dest>`
  succeeds).
- `src/versed/ingest/symbol_pipeline.py`'s `install_into`: add `--` before
  `pip_spec` in the `uv pip install` argv list. Verified working via a real
  install before writing this task.
- `.github/workflows/ci.yml`: add a top-level `permissions: contents: read`
  block.
- `scripts/introspect_worker.py`: update the module docstring to state the
  real trust boundary plainly — importing a package executes that
  package's code with the invoking user's full privileges; the isolated
  venv separates *dependencies*, not *privilege*. Only point this at
  packages you trust.

Verification: `uvx pip-audit --strict -r requirements.txt` output lists
real project package names (not just `pip-audit` itself); `docker compose
up -d db` then `docker compose port db 5432` shows `127.0.0.1:5433`; the
existing `tests/unit/test_docs_fetch.py` and
`tests/integration/test_symbol_pipeline.py` suites still pass with the `--`
change (proves the separator didn't break normal operation); `uv run
pytest tests/unit -v` and `uv run ruff check .` / `format --check .` pass.

---

### Task 16: Final-review fixes — symbol timeline correctness

Addresses two demonstrated-reproducible bugs in `diff.py`'s move-pairing
(the mechanism behind the project's headline "moved" events) and the
introspection worker's silent failure mode.

- **One-to-many move pairing**: `removed_by_short` only checks the removed
  side has exactly one candidate; the added side is unchecked and
  `matched_removed` doesn't prevent a second pairing. Fix: also build
  `added_by_short`, and only pair when BOTH sides have exactly one
  candidate for that short name. Add a test reproducing the exact
  finding: `before={'a.Foo'}`, `after={'b.Foo','c.Foo'}` must NOT produce
  two `moved` events (one of `b.Foo`/`c.Foo` should be a plain `added`
  event instead, and `a.Foo` moved to only the other).
- **Method short-name collision across unrelated classes**: `_short_name`
  on `module.Class.method` is just `"method"`, so unrelated methods named
  the same thing on unrelated classes pair as a false "moved" event. Fix:
  require `before.kind == after.kind` AND, when `kind == "method"`, compare
  the last TWO dot-separated segments (`Class.method`) instead of one, in
  the move-pairing step only (not the existing added/removed/changed
  logic, which already works correctly on full qualified names). Add a
  test reproducing the exact finding: `before={'x.Alpha.invoke'}`,
  `after={'y.Beta.invoke'}` must NOT produce a `moved` event.
- **Silent import failures in `introspect_worker.py`**: `except Exception:
  continue` in the module-walk loop currently has zero signal. Fix: count
  skipped modules and print `f"skipped {n} modules: {names}"` to **stderr**
  (stdout is the JSONL channel, must stay clean) at the end of
  `walk_public_symbols`. Surface the count in `build_symbols_for_version`'s
  return value or echo so a partial introspection is visible, not
  plausible-looking.

Verification: new unit tests in `tests/unit/test_diff.py` reproducing both
scenarios above, both passing; existing `tests/unit/test_diff.py` (6
original tests) still pass unchanged; `tests/unit/test_introspect_worker.py`
extended to confirm a deliberately-broken fake submodule is counted and
reported, not silently dropped; `uv run pytest tests/unit -v` and lint/format
pass.

---

### Task 17: Final-review fixes — venv cache re-runnability

Addresses: Task 12C's own fix (`uv venv --clear`) defeats the incremental-
rerun goal it was written for — `--clear` destroys and fully reinstalls
the venv on every run, so `.versed-venvs/` caches nothing. `uv venv
--allow-existing` ("Preserve any existing files or directories at the
target path") is the flag that actually matches the stated intent, with
`uv pip install` then being a fast near-no-op for an already-satisfied
spec. Verified via `uv venv --help` before writing this task (same source
Task 12C itself verified `--clear` against).

- `src/versed/ingest/symbol_pipeline.py`'s `create_isolated_python`:
  replace `--clear` with `--allow-existing`.
- Update the regression test from Task 12C
  (`test_create_isolated_python_succeeds_on_existing_venv_dir`) — it should
  still pass unchanged, since the fix under test is "succeeds when called
  twice against the same directory," which `--allow-existing` satisfies
  just as well as `--clear` did, for a different (correct) reason.

**Deferred, not in this task — flagged for a real follow-up, not rushed
here:** the final review also found that (a) several integration tests
issue unscoped `DELETE`s against whichever database `DATABASE_URL` points
at, which is the same database the CLI's real pipeline output lives in —
needs a dedicated test-database story (a second docker-compose service or
a `versed_test` database), a genuine infrastructure decision, not a
one-line fix; (b) CI's unit-test job never exercises any DB-backed module,
even though three of the five integration tests need only Postgres (not
Ollama or network) and could run in CI with a service container — needs
someone to verify which specific tests are genuinely Postgres-only first;
(c) `build_manifest()` re-resolves "latest" independently on every CLI
invocation, so a release published mid-pipeline could skew chunks/symbols/
timeline to different actual versions — the real fix (resolve once,
persist, add a `--refresh` or `--version` flag) is a small design decision
that deserves its own attention rather than being folded into a mechanical
parallel-fix batch.

Verification: `uv run pytest tests/integration/test_symbol_pipeline.py -v
-m integration` passes (all 3 tests, including the 12C regression test);
`uv run python -m versed.cli build-symbols` for a single already-cached
version completes fast (proving the cache is actually reused, not just
"doesn't error") — record the actual before/after timing difference.

---

### Task 18: Final-review fixes — ponytail cleanup and debt tracking

Addresses the whole-repo ponytail-audit and ponytail-debt findings from the
completed-plan sweep: unused dependencies, unused generated scaffolding,
and two real deferred-technique decisions that are documented in this plan
but carry no marker at their actual code sites (so a future editor of those
files would never know the missing retry logic is deliberate).

- Remove `langchain-core` and `pytest-cov` from `pyproject.toml`'s
  dependencies — confirmed zero imports anywhere in `src/`, `scripts/`, or
  `tests/` by the audit. Regenerate `uv.lock` (`uv sync --all-groups`).
- Trim `migrations/env.py`'s unused offline-migration code path (Alembic
  generates both an online and offline mode by default; this project only
  ever runs online, against a live `DATABASE_URL`) and any unused default
  config left in `alembic.ini` by `alembic init` that this project doesn't
  read. Do not remove anything actually referenced by the working
  migration flow — verify `uv run alembic upgrade head` and `alembic
  current` still work identically after trimming, not just that the file
  looks shorter.
- Add the two debt markers the audit specifically asked for, at the exact
  code sites the plan's "Deferred production-grade techniques" note
  describes but doesn't mark:
  - `src/versed/ingest/docs_pipeline.py` (near the `embed_documents` call):
    `# ponytail: no retry on embedding calls, add tenacity if a transient
    failure ever kills a real ingestion run`
  - `src/versed/ingest/symbol_pipeline.py` (near the `uv pip install`
    subprocess call): `# ponytail: no retry on package install, add
    tenacity if a transient failure ever kills a real build-symbols run`
- Do NOT touch `src/versed/config.py`'s `Settings.data_dir` field — the
  final review's Task 17 follow-up (deferred, see above) identifies a real
  future use for it (persisting a resolved "latest" version), so removing
  it now would just mean re-adding it later.
- Do NOT change the CI workflow's per-job `uv sync` structure — the audit
  flagged three independent `uv sync` calls as redundant, but Task 13's
  own review explicitly praised this structure for naming failures
  separately in the PR checks list. This is a real, considered tradeoff
  already made deliberately, not an oversight — leave it as-is.

Verification: `git grep -c 'langchain_core\|langchain-core'` returns zero
hits outside `uv.lock`/lockfile metadata; `uv run alembic upgrade head`
against a fresh database still succeeds after the migrations/alembic.ini
trim; `uv run pytest tests/unit -v` and lint/format still pass; both
`ponytail:` comments are present at their specified locations.
