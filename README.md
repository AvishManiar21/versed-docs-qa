# Versed

Version-aware documentation QA over LangChain's docs. See
`docs/superpowers/specs/2026-09-02-version-aware-docs-qa-design.md` for the
full design.

## Setup

    cp .env.example .env   # fill in OPENAI_API_KEY
    uv sync
    docker compose up -d db
    uv run alembic upgrade head
