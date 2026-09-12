import uuid
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from versed.db.models import Symbol, SymbolEvent

Category = Literal[
    "version_explicit", "version_implicit", "migration", "removed_api", "unanswerable"
]


class GoldenQuestion(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    question: str
    category: Category
    target_version: str | None = None
    expected_answer: str | None = None
    expected_symbols: list[str] = Field(default_factory=list)
    should_abstain: bool = False
    reviewed: bool = False
    reviewer_note: str | None = None

    @model_validator(mode="after")
    def _abstain_has_no_answer(self) -> "GoldenQuestion":
        if self.should_abstain and self.expected_answer is not None:
            raise ValueError("should_abstain rows must not carry an expected_answer")
        return self


def write_golden_set(questions: list[GoldenQuestion], path: Path) -> None:
    payload = [q.model_dump(mode="json") for q in questions]
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def read_golden_set(path: Path) -> list[GoldenQuestion]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [GoldenQuestion.model_validate(row) for row in raw]


_OUT_OF_SCOPE_QUESTIONS = [
    "How do I configure a Kubernetes ingress controller?",
    "What's the correct syntax for a Rust match statement?",
    "How do I set up a MySQL replication cluster?",
    "What's the best way to deploy a React Native app to the App Store?",
    "How do I write a device driver for a USB webcam on Linux?",
    "What's the time complexity of quicksort in the worst case?",
    "How do I configure OAuth2 for a Django REST Framework project?",
    "What's the difference between TCP and UDP handshakes?",
    "How do I set up a Terraform module for an AWS VPC?",
    "What's the correct way to use goroutines and channels in Go?",
]

_FABRICATED_SYMBOLS = [
    "langchain.agents.create_supervisor_agent",
    "langchain.chains.SelfCorrectingChain",
    "langchain.memory.QuantumMemory",
    "langchain.tools.AutoBrowserTool",
    "langchain.retrievers.NeuralSymbolicRetriever",
    "langchain.output_parsers.YAMLStrictParser",
    "langchain.agents.load_tools.load_secure_tools",
    "langchain.chains.base.Chain.astream_batch",
    "langchain.embeddings.QuantizedEmbeddings",
    "langchain.callbacks.SlackNotifierCallback",
]


def _short_name(qualified_name: str) -> str:
    return qualified_name.rsplit(".", 1)[-1]


def _find_prior_alternative(qualified_name: str, session: Session) -> str | None:
    """A removed symbol may have been deprecated with a recorded alternative
    at an earlier version transition; diff_versions() never emits a
    `deprecated` event for a name after it's gone, so any match here
    necessarily predates the removal.
    """
    stmt = (
        select(SymbolEvent.detail)
        .where(SymbolEvent.qualified_name == qualified_name)
        .where(SymbolEvent.event_type == "deprecated")
        .where(SymbolEvent.detail.is_not(None))
        .order_by(SymbolEvent.to_version)
        .limit(1)
    )
    return session.scalars(stmt).first()


def _real_events(
    session: Session, versions: list[str], event_types: list[str]
) -> list[SymbolEvent]:
    # ponytail: filters out non-langchain test-fixture rows that share the
    # dev DB (e.g. "pkg.fn"); real per-test DB isolation is still deferred,
    # see the Foundation plan's Global Constraints.
    stmt = (
        select(SymbolEvent)
        .where(SymbolEvent.qualified_name.like("langchain%"))
        .where(SymbolEvent.to_version.in_(versions))
        .where(SymbolEvent.event_type.in_(event_types))
        .order_by(SymbolEvent.qualified_name)
    )
    return list(session.scalars(stmt).all())


def _generate_version_explicit(
    session: Session, versions: list[str], used: set[str], n: int
) -> list[GoldenQuestion]:
    events = _real_events(session, versions, ["added", "deprecated", "changed", "removed"])
    questions: list[GoldenQuestion] = []
    for event in events:
        if len(questions) >= n:
            break
        if event.qualified_name in used:
            continue
        used.add(event.qualified_name)
        short = _short_name(event.qualified_name)
        if event.event_type == "added":
            question = f"Was `{short}` available in LangChain {event.from_version}?"
            answer = f"No — `{event.qualified_name}` was not introduced until {event.to_version}."
            target_version = event.from_version
        elif event.event_type == "deprecated":
            question = f"In version {event.to_version}, is `{short}` still recommended for use?"
            if event.detail:
                answer = f"It's deprecated as of {event.to_version}. Recommended alternative: {event.detail}."
            else:
                answer = f"It's deprecated as of {event.to_version}. No replacement was recorded."
            target_version = event.to_version
        elif event.event_type == "changed":
            question = (
                f"In version {event.to_version}, does `{short}` have the same signature as before?"
            )
            answer = f"No — its signature changed in {event.to_version}: {event.detail}"
            target_version = event.to_version
        else:  # removed
            question = f"Is `{short}` still available in version {event.to_version}?"
            replacement = _find_prior_alternative(event.qualified_name, session)
            tail = (
                f" It was previously deprecated with recommended alternative: {replacement}."
                if replacement
                else ""
            )
            answer = (
                f"No — `{event.qualified_name}` was removed going from "
                f"{event.from_version} to {event.to_version}.{tail}"
            )
            target_version = event.to_version
        questions.append(
            GoldenQuestion(
                question=question,
                category="version_explicit",
                target_version=target_version,
                expected_answer=answer,
                expected_symbols=[event.qualified_name],
            )
        )
    return questions


def _generate_migration(
    session: Session, versions: list[str], used: set[str], n: int
) -> list[GoldenQuestion]:
    events = _real_events(session, versions, ["moved", "changed"])
    questions: list[GoldenQuestion] = []
    for event in events:
        if len(questions) >= n:
            break
        if event.qualified_name in used:
            continue
        used.add(event.qualified_name)
        short = _short_name(event.qualified_name)
        question = (
            f"I'm upgrading from {event.from_version} to {event.to_version} — "
            f"what changed for `{short}`?"
        )
        questions.append(
            GoldenQuestion(
                question=question,
                category="migration",
                target_version=event.to_version,
                expected_answer=event.detail,  # always set for moved/changed, see diff_versions()
                expected_symbols=[event.qualified_name],
            )
        )
    return questions


def _generate_removed_api(
    session: Session, versions: list[str], used: set[str], n: int
) -> list[GoldenQuestion]:
    events = _real_events(session, versions, ["removed"])
    questions: list[GoldenQuestion] = []
    for event in events:
        if len(questions) >= n:
            break
        if event.qualified_name in used:
            continue
        used.add(event.qualified_name)
        short = _short_name(event.qualified_name)
        replacement = _find_prior_alternative(event.qualified_name, session)
        tail = (
            f" It was previously deprecated with recommended alternative: {replacement}."
            if replacement
            else " No replacement was recorded."
        )
        answer = (
            f"`{event.qualified_name}` was removed going from "
            f"{event.from_version} to {event.to_version}.{tail}"
        )
        questions.append(
            GoldenQuestion(
                question=f"What happened to `{short}`? Can I still use it in {event.to_version}?",
                category="removed_api",
                target_version=event.to_version,
                expected_answer=answer,
                expected_symbols=[event.qualified_name],
            )
        )
    return questions


def _generate_version_implicit(
    session: Session, current_version: str, used: set[str], n: int
) -> list[GoldenQuestion]:
    stmt = (
        select(Symbol)
        .where(Symbol.qualified_name.like("langchain%"))
        .where(Symbol.version == current_version)
        .where(Symbol.docstring.is_not(None))
        .order_by(Symbol.qualified_name)
    )
    symbols = session.scalars(stmt).all()
    questions: list[GoldenQuestion] = []
    for symbol in symbols:
        if len(questions) >= n:
            break
        if symbol.qualified_name in used:
            continue
        used.add(symbol.qualified_name)
        short = _short_name(symbol.qualified_name)
        assert symbol.docstring is not None  # filtered by the query above
        summary = symbol.docstring.strip().splitlines()[0][:200]
        questions.append(
            GoldenQuestion(
                question=f"Does LangChain still have `{short}`?",
                category="version_implicit",
                target_version=current_version,
                expected_answer=(
                    f"Yes — `{symbol.qualified_name}` is available in the current "
                    f"version ({current_version}). {summary}"
                ),
                expected_symbols=[symbol.qualified_name],
            )
        )
    return questions


def _generate_unanswerable(session: Session, versions: list[str]) -> list[GoldenQuestion]:
    questions = [
        GoldenQuestion(
            question=text,
            category="unanswerable",
            should_abstain=True,
        )
        for text in _OUT_OF_SCOPE_QUESTIONS
    ]
    for name in _FABRICATED_SYMBOLS:
        exists = session.scalar(
            select(Symbol.id)
            .where(Symbol.qualified_name == name)
            .where(Symbol.version.in_(versions))
        )
        if exists is not None:
            raise ValueError(f"fabricated symbol name {name!r} actually exists now — pick another")
        short = _short_name(name)
        questions.append(
            GoldenQuestion(
                question=f"Does LangChain provide `{short}`?",
                category="unanswerable",
                should_abstain=True,
            )
        )
    return questions


def generate_golden_set(session: Session, versions: list[str]) -> list[GoldenQuestion]:
    """`versions` must be the manifest's version list, ordered oldest to
    newest — the last entry is treated as "current" for version_implicit.
    """
    used: set[str] = set()
    current_version = versions[-1]
    questions: list[GoldenQuestion] = []
    questions += _generate_version_explicit(session, versions, used, n=40)
    questions += _generate_migration(session, versions, used, n=30)
    questions += _generate_removed_api(session, versions, used, n=30)
    questions += _generate_version_implicit(session, current_version, used, n=30)
    questions += _generate_unanswerable(session, versions)
    return questions
