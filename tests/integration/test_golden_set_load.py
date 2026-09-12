import pytest
from sqlalchemy import select

from versed.db.models import EvalQuestion
from versed.db.session import get_session
from versed.golden_set import GoldenQuestion, write_golden_set
from versed.golden_set_load import load_golden_set

pytestmark = pytest.mark.integration


def test_load_golden_set_rejects_unreviewed_rows(tmp_path):
    path = tmp_path / "golden_set.yaml"
    write_golden_set(
        [
            GoldenQuestion(
                question="q1", category="unanswerable", should_abstain=True, reviewed=False
            )
        ],
        path,
    )
    with get_session() as session:
        with pytest.raises(ValueError, match="not marked reviewed"):
            load_golden_set(path, session)


def test_load_golden_set_upserts_reviewed_rows(tmp_path):
    with get_session() as session:
        session.query(EvalQuestion).delete()
        session.commit()

    path = tmp_path / "golden_set.yaml"
    write_golden_set(
        [
            GoldenQuestion(
                question="Is X still valid?",
                category="removed_api",
                target_version="0.3",
                expected_answer="No.",
                expected_symbols=["langchain.x"],
                reviewed=True,
                reviewer_note="looks right",
            )
        ],
        path,
    )

    with get_session() as session:
        count = load_golden_set(path, session)
        assert count == 1
        stored = session.scalars(select(EvalQuestion)).all()
        assert len(stored) == 1
        assert stored[0].human_label == "looks right"
        assert stored[0].expected_symbols == ["langchain.x"]
