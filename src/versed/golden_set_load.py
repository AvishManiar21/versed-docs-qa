from pathlib import Path

from sqlalchemy.orm import Session

from versed.db.models import EvalQuestion
from versed.golden_set import read_golden_set


def load_golden_set(path: Path, session: Session) -> int:
    """Validate every row is reviewed, then replace eval_question wholesale.

    Refuses to load anything if even one row is unreviewed — a golden set
    that's half-loaded is worse than one that's not loaded at all, since
    eval runs would silently score against an incomplete set.
    """
    questions = read_golden_set(path)
    unreviewed = [q for q in questions if not q.reviewed]
    if unreviewed:
        raise ValueError(
            f"{len(unreviewed)} question(s) not marked reviewed: true — "
            f"first unreviewed id is {unreviewed[0].id}"
        )

    session.query(EvalQuestion).delete()
    for q in questions:
        session.add(
            EvalQuestion(
                id=q.id,
                question=q.question,
                category=q.category,
                target_version=q.target_version,
                expected_answer=q.expected_answer,
                expected_symbols=q.expected_symbols,
                should_abstain=q.should_abstain,
                human_label=q.reviewer_note,
            )
        )
    session.commit()
    return len(questions)
