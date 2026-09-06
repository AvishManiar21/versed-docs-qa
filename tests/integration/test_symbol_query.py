import pytest

from versed.db.models import SymbolEvent
from versed.db.session import get_session
from versed.ingest.symbol_query import symbol_history

pytestmark = pytest.mark.integration


def test_symbol_history_matches_by_suffix_and_orders_chronologically():
    with get_session() as session:
        session.query(SymbolEvent).filter(
            SymbolEvent.qualified_name.like("%probe_symbol_query%")
        ).delete()
        session.add(
            SymbolEvent(
                qualified_name="pkg.b.probe_symbol_query",
                from_version="0.2",
                to_version="0.3",
                event_type="added",
                detail=None,
            )
        )
        session.add(
            SymbolEvent(
                qualified_name="pkg.b.probe_symbol_query",
                from_version="0.3",
                to_version="1.0",
                event_type="deprecated",
                detail="pkg.c.new_probe",
            )
        )
        session.commit()

        events = symbol_history("probe_symbol_query", session)
        assert [e.to_version for e in events] == ["0.3", "1.0"]
        assert [e.event_type for e in events] == ["added", "deprecated"]
