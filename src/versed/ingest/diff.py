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


def _move_key(qualified_name: str, kind: str) -> str:
    """Key used only for move-pairing. Methods compare on `Class.method`
    (the last two dot-separated segments) so unrelated methods of the same
    name on unrelated classes don't collide; everything else compares on
    the last segment, as before.
    """
    if kind == "method":
        return ".".join(qualified_name.rsplit(".", 2)[-2:])
    return _short_name(qualified_name)


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

    removed_by_key: dict[tuple[str, str], list[str]] = {}
    for name in removed_names:
        kind = from_symbols[name].kind
        removed_by_key.setdefault((kind, _move_key(name, kind)), []).append(name)

    added_by_key: dict[tuple[str, str], list[str]] = {}
    for name in added_names:
        kind = to_symbols[name].kind
        added_by_key.setdefault((kind, _move_key(name, kind)), []).append(name)

    matched_added: set[str] = set()
    matched_removed: set[str] = set()
    moved_pairs: list[tuple[str, str]] = []
    for added_name in sorted(added_names):
        kind = to_symbols[added_name].kind
        key = (kind, _move_key(added_name, kind))
        removed_candidates = removed_by_key.get(key, [])
        added_candidates = added_by_key.get(key, [])
        if len(removed_candidates) == 1 and len(added_candidates) == 1:
            old_name = removed_candidates[0]
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
