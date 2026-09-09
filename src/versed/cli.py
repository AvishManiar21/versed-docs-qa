from pathlib import Path

import typer

from versed.db.session import get_session
from versed.ingest.docs_pipeline import ingest_docs_for_version
from versed.ingest.manifest import build_manifest
from versed.ingest.symbol_pipeline import build_symbols_for_version
from versed.ingest.symbol_query import symbol_history
from versed.ingest.timeline_pipeline import build_timeline
from versed.rag import build_rag_chain

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


@app.command("ask")
def ask_cmd(question: str, k: int = 5) -> None:
    chain = build_rag_chain(k=k)
    typer.echo(chain.invoke(question))


if __name__ == "__main__":
    app()
