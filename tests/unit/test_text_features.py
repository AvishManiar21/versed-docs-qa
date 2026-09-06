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
