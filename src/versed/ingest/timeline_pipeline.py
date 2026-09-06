from sqlalchemy import select
from sqlalchemy.orm import Session

from versed.db.models import Symbol, SymbolEvent
from versed.ingest.diff import SymbolRow, diff_versions


def _load_symbols(version: str, session: Session) -> dict[str, SymbolRow]:
    rows = session.scalars(select(Symbol).where(Symbol.version == version)).all()
    return {
        row.qualified_name: SymbolRow(
            qualified_name=row.qualified_name,
            kind=row.kind,
            signature=row.signature,
            deprecated_since=row.deprecated_since,
            alternative=row.alternative,
        )
        for row in rows
    }


def build_timeline(ordered_versions: list[str], session: Session) -> int:
    session.query(SymbolEvent).delete()
    total = 0
    for from_version, to_version in zip(ordered_versions, ordered_versions[1:]):
        from_symbols = _load_symbols(from_version, session)
        to_symbols = _load_symbols(to_version, session)
        events = diff_versions(from_version, from_symbols, to_version, to_symbols)
        for event in events:
            session.add(
                SymbolEvent(
                    qualified_name=event.qualified_name,
                    from_version=event.from_version,
                    to_version=event.to_version,
                    event_type=event.event_type,
                    detail=event.detail,
                )
            )
        total += len(events)
    session.commit()
    return total
