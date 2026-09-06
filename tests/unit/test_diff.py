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
    after = {
        "langchain.agents.create_agent": row("langchain.agents.create_agent", sig="(system_prompt)")
    }
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
