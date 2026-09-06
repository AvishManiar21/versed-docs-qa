import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from introspect_worker import parse_deprecation, walk_public_symbols  # noqa: E402

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


def test_broken_submodule_is_counted_and_reported_not_silently_dropped(tmp_path, capsys):
    # Repro from final review: a submodule that raises on import used to be
    # swallowed by a bare `except Exception: continue` with zero signal.
    pkg_dir = tmp_path / "fake_introspect_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "good.py").write_text("def working_fn():\n    pass\n")
    (pkg_dir / "broken.py").write_text("raise RuntimeError('boom')\n")

    sys.path.insert(0, str(tmp_path))
    try:
        skipped_count = walk_public_symbols("fake_introspect_pkg")
    finally:
        sys.path.remove(str(tmp_path))
        for name in list(sys.modules):
            if name == "fake_introspect_pkg" or name.startswith("fake_introspect_pkg."):
                del sys.modules[name]

    captured = capsys.readouterr()

    # The good submodule's symbol still made it to stdout (the JSONL channel).
    assert "working_fn" in captured.out

    # The broken submodule is counted and reported on stderr, not silently
    # dropped, and stdout stays clean JSONL (no report text on stdout).
    assert skipped_count == 1
    assert "skipped 1 modules" in captured.err
    assert "fake_introspect_pkg.broken" in captured.err
    assert "skipped" not in captured.out
