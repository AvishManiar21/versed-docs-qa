from sqlalchemy import select
from sqlalchemy.orm import Session

from versed.db.models import SymbolEvent


def symbol_history(qualified_name_suffix: str, session: Session) -> list[SymbolEvent]:
    """Return every event whose qualified_name ends with the given suffix,
    ordered chronologically. Suffix match lets callers query by short name
    (e.g. "create_agent") without knowing the full module path.
    """
    stmt = (
        select(SymbolEvent)
        .where(SymbolEvent.qualified_name.like(f"%{qualified_name_suffix}"))
        .order_by(SymbolEvent.to_version)
    )
    return list(session.scalars(stmt).all())
