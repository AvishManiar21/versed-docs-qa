# tests/integration/test_timeline_pipeline.py
import pytest
from sqlalchemy import select

from versed.db.models import Symbol, SymbolEvent
from versed.db.session import get_session
from versed.ingest.timeline_pipeline import build_timeline

pytestmark = pytest.mark.integration


def test_build_timeline_across_three_synthetic_versions():
    with get_session() as session:
        session.query(SymbolEvent).delete()
        session.query(Symbol).filter(Symbol.version.in_(["t1", "t2", "t3"])).delete()

        session.add(
            Symbol(version="t1", qualified_name="pkg.old_fn", kind="function", signature="()")
        )
        session.add(
            Symbol(
                version="t2",
                qualified_name="pkg.old_fn",
                kind="function",
                signature="()",
                deprecated_since="t2",
                alternative="pkg.new_fn",
            )
        )
        session.add(
            Symbol(version="t3", qualified_name="pkg.new_fn", kind="function", signature="()")
        )
        session.commit()

        total = build_timeline(["t1", "t2", "t3"], session)
        assert total == 3

        events = session.scalars(
            select(SymbolEvent).where(SymbolEvent.qualified_name.like("%old_fn%"))
        ).all()
        event_types = {e.event_type for e in events}
        assert "deprecated" in event_types
