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
