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
