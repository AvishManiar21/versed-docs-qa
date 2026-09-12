# Versed

**Version-aware documentation QA over LangChain's docs.**

Ask a documentation chatbot "how do I build an agent in LangChain?" and most
will confidently hand you `create_react_agent` — an API that was renamed in
v1.0. Standard RAG has no concept of *when* a document was true, so it treats
a page from 2023 and a page from today as equally valid evidence. Versed adds
that missing dimension: every answer is resolved against a specific library
version, backed by a symbol timeline built by introspecting the actual
installed packages, not by hoping the model notices a version number in a
paragraph.

Full problem statement, architecture, and evaluation design:
[`docs/superpowers/specs/2026-09-02-version-aware-docs-qa-design.md`](docs/superpowers/specs/2026-09-02-version-aware-docs-qa-design.md).

## Project status

Actively in development. See [open milestones](../../milestones) for
phase-by-phase progress and [issues](../../issues) for individual tasks.

| Phase | Status |
|---|---|
| 1. Foundation — corpus ingestion & symbol timeline | ✅ done |
| 2. Baseline RAG & golden eval set | ✅ done |
| 3. Real system — hybrid retrieval & LangGraph pipeline | ⬜ not started |
| 4. Evaluation harness & CI quality gate | ⬜ not started |
| 5. Guardrails & adversarial testing | ⬜ not started |
| 6. Production ops & deployment | ⬜ not started |
| 7. Extracted `versed-eval` library | ⬜ not started |
| 8. Publication (datasets, writeup) | ⬜ not started |

Each phase has its own implementation plan under
[`docs/superpowers/plans/`](docs/superpowers/plans/).

## Repository layout

```
versed-docs-qa/
├── src/versed/            # application code
│   ├── config.py          # settings (env-driven)
│   ├── db/                # SQLAlchemy models + session
│   ├── ingest/             # docs & symbol ingestion pipelines
│   └── cli.py              # command-line entry point
├── scripts/
│   └── introspect_worker.py  # stdlib-only script run inside isolated
│                              # per-version venvs — deliberately has zero
│                              # dependency on src/versed
├── migrations/              # Alembic schema migrations
├── tests/
│   ├── unit/                # fast, no external services
│   └── integration/         # needs Postgres (`docker compose up -d db`);
│                             # some are also marked `slow` (real network)
├── docs/
│   └── superpowers/
│       ├── specs/           # design specs — the "why" and "what"
│       └── plans/           # implementation plans — the "how", task by task
├── .github/
│   ├── ISSUE_TEMPLATE/       # one issue per plan task
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/ci.yml
├── docker-compose.yml        # Postgres + pgvector, local dev only
└── pyproject.toml
```

New files go in the directory that matches their responsibility above —
see [CONTRIBUTING.md](CONTRIBUTING.md) if you're unsure where something
belongs.

## Setup

### Prerequisites

| Tool | Why | Install |
|---|---|---|
| **Python 3.12+** | runtime | see OS instructions below |
| **[uv](https://docs.astral.sh/uv/)** | dependency & venv management (also fetches Python 3.11 for the isolated introspection venvs) | see OS instructions below |
| **Docker** | runs Postgres + pgvector locally | see OS instructions below |
| **git** | version control | usually preinstalled |
| **[Ollama](https://ollama.com)** | local embeddings (`nomic-embed-text`) | `curl -fsSL https://ollama.com/install.sh \| sh` then `ollama pull nomic-embed-text` |

<details>
<summary><b>macOS</b></summary>

```bash
# Homebrew (skip if already installed): https://brew.sh
brew install uv git
brew install --cask docker   # then launch Docker.app once to finish setup
```

</details>

<details>
<summary><b>Ubuntu / Debian Linux</b></summary>

```bash
sudo apt update && sudo apt install -y git

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Docker Engine + Compose plugin
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"   # log out/in (or `newgrp docker`) after this
```

</details>

<details>
<summary><b>Windows</b></summary>

Run in **PowerShell**:

```powershell
winget install --id Astral-SH.Uv -e
winget install --id Docker.DockerDesktop -e
winget install --id Git.Git -e
```

Launch Docker Desktop once and make sure it's running before continuing.
WSL2 backend is required and is enabled automatically by Docker Desktop's
installer if it isn't already.

All commands below are identical in PowerShell and in WSL2/Git Bash.

</details>

### Get it running

```bash
git clone https://github.com/AvishManiar21/versed-docs-qa.git
cd versed-docs-qa

cp .env.example .env

uv sync                        # installs dependencies into .venv/
uv run pre-commit install      # runs ruff automatically on every commit
docker compose up -d db        # starts Postgres+pgvector on localhost:5433
uv run alembic upgrade head    # creates the schema
```

Verify the database is up:

```bash
docker compose ps    # "db" should show (healthy)
```

### Usage

```bash
# Ingest LangChain's docs across every tracked version
uv run python -m versed.cli ingest-docs

# Build the symbol timeline (spins up isolated per-version venvs — slow, first run)
uv run python -m versed.cli build-symbols
uv run python -m versed.cli build-timeline

# Ask what changed for a given API
uv run python -m versed.cli timeline create_agent
```

### Running tests

```bash
uv run pytest tests/unit -v                     # fast, no external services
uv run pytest tests/integration -v -m integration   # needs `docker compose up -d db`
```

Some integration tests are also marked `slow` (they hit the real network —
git clones, pip installs, Ollama embedding calls). Run everything, slow
included, with:

```bash
uv run pytest tests/integration -v -m integration
```

## Contributing

This repo is developed issue → branch → PR, one task per plan at a time.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.

## License

[MIT](LICENSE)
