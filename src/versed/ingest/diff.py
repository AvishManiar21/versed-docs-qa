from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolRow:
    qualified_name: str
    kind: str
    signature: str
    deprecated_since: str | None
    alternative: str | None


@dataclass(frozen=True)
class DiffEvent:
    qualified_name: str
    from_version: str | None
    to_version: str
    event_type: str
    detail: str | None


def _short_name(qualified_name: str) -> str:
    return qualified_name.rsplit(".", 1)[-1]


def diff_versions(
    from_version: str,
    from_symbols: dict[str, SymbolRow],
    to_version: str,
    to_symbols: dict[str, SymbolRow],
) -> list[DiffEvent]:
    events: list[DiffEvent] = []

    removed_names = set(from_symbols) - set(to_symbols)
    added_names = set(to_symbols) - set(from_symbols)
    common_names = set(from_symbols) & set(to_symbols)

    removed_by_short: dict[str, list[str]] = {}
    for name in removed_names:
        removed_by_short.setdefault(_short_name(name), []).append(name)

    matched_added: set[str] = set()
    matched_removed: set[str] = set()
    moved_pairs: list[tuple[str, str]] = []
    for added_name in sorted(added_names):
        candidates = removed_by_short.get(_short_name(added_name), [])
        if len(candidates) == 1:
            old_name = candidates[0]
            moved_pairs.append((old_name, added_name))
            matched_added.add(added_name)
            matched_removed.add(old_name)

    for old_name, new_name in moved_pairs:
        events.append(
            DiffEvent(
                qualified_name=new_name,
                from_version=from_version,
                to_version=to_version,
                event_type="moved",
                detail=f"Moved from {old_name} to {new_name}",
            )
        )

    for name in sorted(added_names - matched_added):
        events.append(
            DiffEvent(
                qualified_name=name,
                from_version=from_version,
                to_version=to_version,
                event_type="added",
                detail=None,
            )
        )

    for name in sorted(removed_names - matched_removed):
        events.append(
            DiffEvent(
                qualified_name=name,
                from_version=from_version,
                to_version=to_version,
                event_type="removed",
                detail=None,
            )
        )

    for name in sorted(common_names):
        before = from_symbols[name]
        after = to_symbols[name]
        if after.deprecated_since and not before.deprecated_since:
            events.append(
                DiffEvent(
                    qualified_name=name,
                    from_version=from_version,
                    to_version=to_version,
                    event_type="deprecated",
                    detail=after.alternative,
                )
            )
        elif before.signature != after.signature:
            events.append(
                DiffEvent(
                    qualified_name=name,
                    from_version=from_version,
                    to_version=to_version,
                    event_type="changed",
                    detail=f"{before.signature} -> {after.signature}",
                )
            )

    return events
