import pytest

from versed.golden_set import GoldenQuestion, read_golden_set, write_golden_set


def test_yaml_round_trip_preserves_all_fields(tmp_path):
    question = GoldenQuestion(
        question="Is X still valid?",
        category="removed_api",
        target_version="0.3",
        expected_answer="No.",
        expected_symbols=["langchain.x"],
        reviewed=True,
        reviewer_note="looks right",
    )
    path = tmp_path / "golden_set.yaml"
    write_golden_set([question], path)
    loaded = read_golden_set(path)
    assert loaded == [question]


def test_should_abstain_rejects_a_non_null_expected_answer():
    with pytest.raises(ValueError, match="should_abstain"):
        GoldenQuestion(
            question="What's the weather?",
            category="unanswerable",
            should_abstain=True,
            expected_answer="This should not be allowed",
        )
