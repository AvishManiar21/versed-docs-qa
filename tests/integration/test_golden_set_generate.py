import pytest

from versed.db.session import get_session
from versed.golden_set import generate_golden_set

pytestmark = pytest.mark.integration

_VERSIONS = ["0.1", "0.2", "0.3", "1.4.0"]


def test_generate_golden_set_produces_150_across_five_categories():
    with get_session() as session:
        questions = generate_golden_set(session, _VERSIONS)

    assert len(questions) == 150
    counts: dict[str, int] = {}
    for q in questions:
        counts[q.category] = counts.get(q.category, 0) + 1
    assert counts == {
        "version_explicit": 40,
        "migration": 30,
        "removed_api": 30,
        "version_implicit": 30,
        "unanswerable": 20,
    }

    unanswerable = [q for q in questions if q.category == "unanswerable"]
    assert all(q.should_abstain and q.expected_answer is None for q in unanswerable)

    non_abstain = [q for q in questions if not q.should_abstain]
    names = [q.expected_symbols[0] for q in non_abstain]
    assert len(names) == len(set(names))  # no symbol reused across categories
    assert all(name.startswith("langchain") for name in names)
