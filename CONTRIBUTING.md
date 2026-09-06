# Contributing

This is a solo portfolio project, but it's run with the same discipline as a
real team project: every unit of work is an issue, every issue is built on
its own branch, every branch merges through a pull request. The point is
partly hygiene and partly the point itself — a public commit/issue/PR history
showing incremental, well-described work is part of what this project is
for.

## Workflow

1. **Pick up an issue.** Every issue corresponds to exactly one task in a
   plan under `docs/superpowers/plans/`. The issue body links to the task's
   section of the plan — that section has the exact files to touch, the code,
   and the verification steps. Don't start a task that has no issue yet;
   file one first (see "Filing an issue" below).

2. **Branch from `main`:**

   ```bash
   git checkout main && git pull
   git checkout -b <issue-number>-<short-slug>
   ```

   Example: `12-db-models-migration` for issue #12. The number makes the
   branch traceable back to its issue at a glance.

3. **Follow the plan's own step sequence** (write test → run it, confirm it
   fails → implement → run it, confirm it passes → commit). Commit at each
   plan-designated commit point, not once at the end — the commit history is
   part of what a reader evaluates.

4. **Open a PR** against `main` using the PR template
   (`.github/PULL_REQUEST_TEMPLATE.md`, filled in automatically). Reference
   the issue with `Closes #<N>` so it closes automatically on merge. The PR
   description's "Test plan" section should show the actual commands run and
   their actual output — not "tests pass," the real terminal output.

5. **Self-review before requesting approval.** Run the task's own
   verification steps plus a code-review pass, and post the findings as a PR
   comment (what was checked, what was fixed, what's left). This is the
   quality gate before a human looks at it — a PR shouldn't be handed over
   with known issues still in it.

6. **Get final approval, then merge via squash merge**, so `main` gets one
   clean commit per task while the branch keeps its granular TDD history for
   anyone who opens the PR later. Branches are kept after merge, not deleted
   — the per-task history stays browsable on GitHub.

## Filing an issue

Use the "Plan task" issue template. Every field maps directly to a section
of the task in its plan document — copy from there, don't re-derive it:

- **Plan / task:** which plan file and which numbered task (e.g.
  `2026-09-03-versed-foundation.md`, Task 7)
- **Goal:** the one-line deliverable
- **Files:** exact paths from the task's `Files:` block
- **Acceptance criteria:** the task's own verification steps (the `Run:` /
  `Expected:` pairs)

File one issue per task, not one issue per plan — a plan's tasks are meant to
be reviewed and merged independently.

## Commit messages

Follow the plan's own commit message for that step where one is given (the
plans specify exact messages for exact commits). Where you're committing
something the plan didn't script, use a Conventional Commits–style prefix:
`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`.

## Branch naming

`<issue-number>-<short-kebab-slug>`, e.g. `7-docs-ingestion-pipeline`. No
`feat/`/`fix/` prefixes — the issue number is the identifier that matters
here.

## Quality bar

Every task review checks two things beyond "does it work": spec compliance
and code quality. Code quality explicitly includes:

- **Prefer a library over hand-rolled code whenever one genuinely fits** —
  don't write what `tiktoken`, `httpx`, `pip-audit`, CodeQL, or `ruff format`
  already do correctly. The one deliberate exception is
  `scripts/introspect_worker.py`, which must stay stdlib-only by design (see
  the plan's Global Constraints).
- **Automated security and dependency scanning runs in CI on every PR** —
  CodeQL (SAST), `pip-audit` (known-vulnerability scan), and Dependabot
  (automated update PRs for Python deps and GitHub Actions versions). See
  `.github/workflows/codeql.yml`, `.github/workflows/ci.yml`, and
  `.github/dependabot.yml`.
- **This is in addition to, not instead of, human/subagent review** — CI
  catches known vulnerability patterns and stale dependencies; it does not
  catch design-level issues like trust-boundary mistakes or a task
  contradicting an already-committed file. Once this plan's tasks are all
  merged, a full-scope review runs both `superpowers:requesting-code-review`
  and the `security-review` skill over everything changed, not just the
  automated per-PR checks.

## Ponytail (anti-over-engineering)

The `ponytail` Claude Code plugin is installed and used as follows — each
skill has one clear point in the process, not a blanket "run everything
always":

- **`ponytail` (main mode)** — always active during implementation. Every
  task's code follows its ladder by default: does this need to exist,
  reuse what's already here, stdlib before custom code, fewest lines that
  work. No separate invocation needed; it governs how code gets written.
- **`ponytail-review`** — run as an additional lens at every task-level
  review and at the final full-scope review, alongside the existing
  spec/quality and `security-review` passes. Hunts specifically for
  reinvented stdlib, unneeded dependencies, and speculative abstractions —
  complements, doesn't replace, correctness-focused review.
- **`ponytail-audit`** and **`ponytail-debt`** — run together, once per
  completed plan/milestone (a whole-repo sweep, not diff-scoped, so too
  coarse for per-task use). `ponytail-audit` finds what to delete or
  simplify repo-wide; `ponytail-debt` harvests any `ponytail:` shortcut
  comments left behind so a deliberate simplification gets tracked instead
  of forgotten.
- **`ponytail-gain`** and **`ponytail-help`** — on-demand only, when asked.

## Where does this file go?

If you're adding something and unsure where it belongs, match it to the
directory it changes together with, per `README.md`'s repository layout:

- Application logic → `src/versed/<module>/`
- Anything that must run with zero third-party dependencies (per-version
  introspection) → `scripts/`, never `src/versed/`
- Schema changes → a new file in `migrations/versions/`, never edit an
  applied migration in place
- Tests mirror the module they test: `tests/unit/test_<module>.py` or
  `tests/integration/test_<module>.py`
- Design rationale (the "why") → `docs/superpowers/specs/`
- Task-by-task build instructions (the "how") → `docs/superpowers/plans/`

Don't invent new top-level directories without updating the layout diagram in
`README.md` in the same PR.
