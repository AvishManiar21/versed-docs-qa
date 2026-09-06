# Task 16: Final-review fixes — symbol timeline correctness

## What was implemented

### Bug 1: one-to-many move pairing (`src/versed/ingest/diff.py`)

Original code built `removed_by_short` (short-name -> list of removed
candidates) and only checked that the **removed** side had exactly one
candidate before pairing an added symbol to it as "moved". It never checked
how many added symbols shared that short name, and `matched_removed` was
tracked but never consulted when looking up candidates — so a single removed
symbol could be paired as "moved" into **every** added symbol sharing its
short name.

Repro: `before={"pkg.a.Foo"}`, `after={"pkg.b.Foo", "pkg.c.Foo"}` produced
**two** `moved` events, both claiming `pkg.a.Foo` moved — once to
`pkg.b.Foo` and once to `pkg.c.Foo`.

Fix: also build `added_by_key` (mirroring `removed_by_key`), and only pair
when **both** sides have exactly one candidate for that key. With two
added candidates sharing the short name, neither pairs: `pkg.a.Foo` stays
`removed`, `pkg.b.Foo` and `pkg.c.Foo` both become `added`.

### Bug 2: method short-name collision across unrelated classes (`src/versed/ingest/diff.py`)

`_short_name` on `module.Class.method` returned just `"method"`, so two
unrelated methods with the same name on unrelated classes (e.g.
`x.Alpha.invoke` and `y.Beta.invoke`) paired as a false "moved" event.

Fix: added `_move_key(qualified_name, kind)` — for `kind == "method"` it
compares the last **two** dot-separated segments (`Class.method`) instead of
one; everything else still uses the last segment. Pairing keys are now
`(kind, move_key)` tuples, so pairing also requires `before.kind ==
after.kind` (both fixes share the same lookup-key change). This is scoped to
the move-pairing step only — the added/removed/changed logic elsewhere in
`diff_versions` still compares full qualified names and is untouched.

### Bug 3: silent import failures (`scripts/introspect_worker.py`)

`except Exception: continue` in the `pkgutil.walk_packages` loop inside
`walk_public_symbols` silently dropped any submodule that failed to import,
with zero signal that introspection was incomplete.

Fix: `walk_public_symbols` now collects the names of skipped modules in a
list, and at the end — if any were skipped — prints
`f"skipped {len(skipped)} modules: {skipped}"` to **stderr** (stdout stays
clean JSONL, unaffected). It also now returns `len(skipped)` so a caller
(e.g. `build_symbols_for_version` in `symbol_pipeline.py`, which is outside
this task's file scope) can additionally surface the count via its own
return value or logging without further changes to `introspect_worker.py`.

## TDD evidence

### Fix 1 + Fix 2 (`tests/unit/test_diff.py`)

RED (before fix, run against original `diff.py`):

```
FAILED tests/unit/test_diff.py::test_one_to_many_short_name_does_not_produce_two_moved_events
  AssertionError: a single removed symbol must not move to two places
  assert 2 < 2
   +  where 2 = len([DiffEvent(qualified_name='pkg.b.Foo', ..., event_type='moved', detail='Moved from pkg.a.Foo to pkg.b.Foo'),
                      DiffEvent(qualified_name='pkg.c.Foo', ..., event_type='moved', detail='Moved from pkg.a.Foo to pkg.c.Foo')])

FAILED tests/unit/test_diff.py::test_method_short_name_collision_across_unrelated_classes_not_paired
  AssertionError: assert {('y.Beta.invoke', 'moved')} == {('x.Alpha.invoke', 'removed'), ('y.Beta.invoke', 'added')}
2 failed, 6 passed in 0.08s
```

GREEN (after fix):

```
tests/unit/test_diff.py::test_detects_added_symbol PASSED
tests/unit/test_diff.py::test_detects_removed_symbol PASSED
tests/unit/test_diff.py::test_detects_newly_deprecated_symbol PASSED
tests/unit/test_diff.py::test_detects_changed_signature PASSED
tests/unit/test_diff.py::test_pairs_rename_into_single_moved_event PASSED
tests/unit/test_diff.py::test_unchanged_symbol_produces_no_event PASSED
tests/unit/test_diff.py::test_one_to_many_short_name_does_not_produce_two_moved_events PASSED
tests/unit/test_diff.py::test_method_short_name_collision_across_unrelated_classes_not_paired PASSED
8 passed in 0.02s
```

### Fix 3 (`tests/unit/test_introspect_worker.py`)

RED (before fix):

```
FAILED tests/unit/test_introspect_worker.py::test_broken_submodule_is_counted_and_reported_not_silently_dropped
  assert skipped_count == 1
  assert None == 1
1 failed, 5 passed in 0.09s
```

GREEN (after fix):

```
tests/unit/test_introspect_worker.py::test_parses_real_langchain_deprecation_notice PASSED
tests/unit/test_introspect_worker.py::test_parses_auto_generated_double_backtick_form PASSED
tests/unit/test_introspect_worker.py::test_parses_package_prefixed_since PASSED
tests/unit/test_introspect_worker.py::test_returns_none_for_non_deprecated_docstring PASSED
tests/unit/test_introspect_worker.py::test_returns_none_for_empty_docstring PASSED
tests/unit/test_introspect_worker.py::test_broken_submodule_is_counted_and_reported_not_silently_dropped PASSED
6 passed in 0.57s
```

The new test builds a real temp package (`good.py` importable, `broken.py`
raises `RuntimeError` at import time), calls `walk_public_symbols` on it, and
asserts (via `capsys`): `working_fn` (from the good module) reached stdout,
`skipped 1 modules` and the broken module's dotted name appeared on stderr,
and the literal string `"skipped"` never appears on stdout.

## Full test suite output (both files, run together as `tests/unit`)

```
collected 27 items

tests/unit/test_chunker.py::test_splits_on_headings_and_tracks_path PASSED
tests/unit/test_chunker.py::test_never_splits_inside_fenced_code_block PASSED
tests/unit/test_chunker.py::test_no_heading_produces_single_unheaded_chunk PASSED
tests/unit/test_config.py::test_settings_reads_database_url_from_env PASSED
tests/unit/test_config.py::test_settings_has_sane_defaults PASSED
tests/unit/test_diff.py::test_detects_added_symbol PASSED
tests/unit/test_diff.py::test_detects_removed_symbol PASSED
tests/unit/test_diff.py::test_detects_newly_deprecated_symbol PASSED
tests/unit/test_diff.py::test_detects_changed_signature PASSED
tests/unit/test_diff.py::test_pairs_rename_into_single_moved_event PASSED
tests/unit/test_diff.py::test_unchanged_symbol_produces_no_event PASSED
tests/unit/test_diff.py::test_one_to_many_short_name_does_not_produce_two_moved_events PASSED
tests/unit/test_diff.py::test_method_short_name_collision_across_unrelated_classes_not_paired PASSED
tests/unit/test_docs_fetch.py::test_fetch_ref_checks_out_tag PASSED
tests/unit/test_docs_fetch.py::test_iter_doc_files_only_returns_subpath_md_and_mdx PASSED
tests/unit/test_introspect_worker.py::test_parses_real_langchain_deprecation_notice PASSED
tests/unit/test_introspect_worker.py::test_parses_auto_generated_double_backtick_form PASSED
tests/unit/test_introspect_worker.py::test_parses_package_prefixed_since PASSED
tests/unit/test_introspect_worker.py::test_returns_none_for_non_deprecated_docstring PASSED
tests/unit/test_introspect_worker.py::test_returns_none_for_empty_docstring PASSED
tests/unit/test_introspect_worker.py::test_broken_submodule_is_counted_and_reported_not_silently_dropped PASSED
tests/unit/test_manifest.py::test_resolve_latest_langchain_version PASSED
tests/unit/test_manifest.py::test_build_manifest_has_four_versions PASSED
tests/unit/test_manifest.py::test_only_current_version_has_extra_pip_specs PASSED
tests/unit/test_text_features.py::test_extracts_inline_code_identifiers PASSED
tests/unit/test_text_features.py::test_ignores_non_identifier_inline_code PASSED
tests/unit/test_text_features.py::test_count_tokens_is_positive_and_roughly_proportional PASSED

27 passed in 0.32s
```

`uv run ruff check` and `uv run ruff format --check` on all 4 changed files:
clean. `uv run mypy src/versed/ingest/diff.py scripts/introspect_worker.py`:
"Success: no issues found in 2 source files".

## Files changed

- `src/versed/ingest/diff.py` — added `_move_key`, replaced
  `removed_by_short`/candidate-count logic with symmetric
  `removed_by_key`/`added_by_key` keyed on `(kind, move_key)`, requiring
  both sides to have exactly one candidate.
- `scripts/introspect_worker.py` — `walk_public_symbols` now tracks and
  reports skipped (unimportable) submodules to stderr and returns the
  skipped count.
- `tests/unit/test_diff.py` — `row()` helper gained an optional `kind`
  parameter (default `"function"`, all existing calls unchanged); added two
  new tests reproducing the exact repro cases from the plan.
- `tests/unit/test_introspect_worker.py` — added one new test with a
  temp on-disk package containing a working submodule and a submodule that
  raises on import, asserting the failure is counted/reported on stderr
  and stdout stays clean.

## Self-review

- New one-to-many pairing test: confirmed RED then GREEN (see TDD evidence
  above) — yes, it reproduces the exact plan repro and required the fix.
- New method-collision test: confirmed RED then GREEN — yes.
- All 6 original `test_diff.py` tests: still pass, **unchanged** (no test
  bodies were edited, only the shared `row()` helper gained an optional
  kwarg with a default that preserves every existing call).
- New introspect_worker behavior: writes exclusively to `sys.stderr`;
  verified via `capsys` that the literal string `"skipped"` never appears
  in `captured.out`, and stdout still carries the good module's JSONL
  symbol line.
- No stray output, warnings, or leftover files: `git status` shows only the
  4 intended modified files; `ruff check`/`ruff format --check`/`mypy` all
  clean; the temp package created by the new introspect_worker test lives
  entirely under pytest's `tmp_path` fixture (auto-cleaned) and the test
  removes its `sys.path` entry and `sys.modules` entries in a `finally`
  block so it can't leak into other tests in the same process.
- Scope check: did not touch `src/versed/ingest/symbol_pipeline.py` or any
  file outside `diff.py`/`introspect_worker.py`/their unit tests, per the
  task's file-scope restriction. The plan's suggestion to also "surface the
  count in `build_symbols_for_version`'s return value or echo" is satisfied
  on the `introspect_worker.py` side by the stderr echo plus the new
  non-None return value from `walk_public_symbols`; wiring that further
  into `symbol_pipeline.py`'s return value was out of this task's declared
  file scope and was not attempted.

## Concerns

None. All three fixes have RED→GREEN TDD evidence, the 6 original
`test_diff.py` tests are byte-for-byte unchanged in body, lint/format/mypy
are clean, and there is no stray output or leftover state.

## Worktree / branch

- Worktree path: `/home/taj/project_1/.claude/worktrees/agent-aaacd31b278360625`
- Branch: `16-diff-correctness` (branched from `origin/15-security-fixes`)
