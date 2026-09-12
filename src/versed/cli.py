from pathlib import Path

import typer

from versed.db.session import get_session
from versed.golden_set import generate_golden_set, read_golden_set, write_golden_set
from versed.golden_set_load import load_golden_set
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
def ask_cmd(question: str, k: int = typer.Option(5, min=1, max=50)) -> None:
    chain = build_rag_chain(k=k)
    typer.echo(chain.invoke(question))


golden_set_app = typer.Typer()
app.add_typer(golden_set_app, name="golden-set")


@golden_set_app.command("generate")
def golden_set_generate_cmd(out: Path = Path("data/golden_set.yaml")) -> None:
    manifest = build_manifest()
    versions = [source.version for source in manifest]
    out.parent.mkdir(parents=True, exist_ok=True)
    with get_session() as session:
        questions = generate_golden_set(session, versions)
    write_golden_set(questions, out)
    typer.echo(f"Wrote {len(questions)} draft questions to {out}")


@golden_set_app.command("load")
def golden_set_load_cmd(path: Path = Path("data/golden_set.yaml")) -> None:
    with get_session() as session:
        count = load_golden_set(path, session)
    typer.echo(f"Loaded {count} questions into eval_question")


@golden_set_app.command("show")
def golden_set_show_cmd(identifier: str, path: Path = Path("data/golden_set.yaml")) -> None:
    """Print one golden-set row plus the real timeline for its symbols, so
    reviewing a question and checking its evidence doesn't require manually
    cross-referencing the YAML file against `versed timeline`.
    """
    questions = read_golden_set(path)

    match = None
    if identifier.isdigit() and int(identifier) < len(questions):
        match = questions[int(identifier)]
    else:
        match = next((q for q in questions if str(q.id) == identifier), None)
    if match is None:
        typer.echo(f"No question found for '{identifier}' (tried index and id)")
        raise typer.Exit(code=1)

    typer.echo(f"[{match.category}] {match.question}")
    if match.target_version:
        typer.echo(f"target_version: {match.target_version}")
    if match.should_abstain:
        typer.echo("expected: should abstain (no ground-truth answer)")
    else:
        typer.echo(f"expected_answer: {match.expected_answer}")

    if not match.expected_symbols:
        typer.echo("(no symbols to check — unanswerable/out-of-scope question)")
        return

    with get_session() as session:
        for symbol in match.expected_symbols:
            suffix = ".".join(symbol.rsplit(".", 2)[-2:]) if symbol.count(".") >= 2 else symbol
            typer.echo(f"\ntimeline for {symbol}:")
            events = symbol_history(suffix, session)
            if not events:
                typer.echo("  (no history found)")
            for event in events:
                arrow = (
                    f"{event.from_version} -> {event.to_version}"
                    if event.from_version
                    else event.to_version
                )
                detail = f" ({event.detail})" if event.detail else ""
                typer.echo(f"  [{arrow}] {event.event_type}: {event.qualified_name}{detail}")


if __name__ == "__main__":
    app()
