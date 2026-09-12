from pathlib import Path

from versed.golden_set import read_golden_set

_PATH = Path(__file__).resolve().parents[2] / "data" / "golden_set.yaml"


def test_committed_golden_set_parses_and_has_expected_category_counts():
    questions = read_golden_set(_PATH)
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

    question_texts = [q.question for q in questions]
    assert len(question_texts) == len(set(question_texts))
