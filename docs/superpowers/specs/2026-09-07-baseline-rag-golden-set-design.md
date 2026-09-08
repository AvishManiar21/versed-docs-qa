# Baseline RAG + Golden Set v1 — Design

Covers spec Milestones 3 and 4 from
`2026-09-02-version-aware-docs-qa-design.md` section 12. Builds on the
completed Foundation plan (corpus ingestion + symbol timeline, Milestones 1-2,
fully merged to `main`).

## 1. Purpose

Milestone 3 needs "naive config end to end, so there is something to beat."
Milestone 4 needs "150 questions, hand-reviewed, in the repo." Together they
produce the first working, demonstrable slice of the actual QA product (not
just ingestion pipelines), plus the fixed question set that every later
milestone's improvements get measured against.

## 2. Scope

**In scope:**
- A CLI command that answers a question via plain dense retrieval + local LLM
  generation, with no version awareness.
- A generator that produces 150 golden-set questions, all mechanically
  drafted from real data (`symbol_event`, `symbol`), for human review.
- The `eval_question` table (schema already fixed by the parent spec, section
  7) and a loader that pushes the reviewed file into it.

**Out of scope (explicitly deferred to later milestones):**
- Hybrid/version-filtered retrieval, reranking, the LangGraph pipeline,
  citations, abstention logic — Milestone 5.
- Scoring anything: recall@k, MRR, faithfulness/correctness judges,
  stale-answer rate, the ablation matrix, CI eval gate, judge calibration —
  all Milestone 6. This plan produces the golden set; it does not grade
  against it.
- RAGAS — planned for Milestone 6's metrics harness (see project memory
  `versed-ragas-milestone-6`), not used here. Golden-set generation stays
  fully mechanical (no LLM in the ground-truth path) to avoid the
  hallucination risk an LLM-drafted testset would carry.
- FastAPI service, guardrails, Redis caching, demo UI — Milestones 7-8.
- The two items already deferred from the Foundation plan's final review
  (version-skew across CLI commands re-resolving "latest" independently, and
  test-database isolation) are not addressed here — this plan reads from and
  writes to the same DB conventions Foundation established, and doesn't
  re-run the version-resolution path in a way that makes the skew risk worse.

## 3. Generation LLM

Ollama, local, CPU-only, no API key, no network dependency: **Phi-3.5-mini**
(3.8B, ~2.2GB). Chosen over a local 7B model (RAM headroom on this machine is
inconsistent) and over an OpenRouter free-tier model (avoids adding a network
dependency and rate-limit ceiling for what's meant to be a weak, beatable
baseline). Revisit for Milestone 5, where generation quality matters more for
the demo.

Embeddings continue to use the already-integrated `nomic-embed-text`
(unchanged from Foundation).

## 4. Baseline RAG architecture

```
versed ask "<question>" [--k 5]

  question
     │
     ▼
  embed (nomic-embed-text, same model used at ingest)
     │
     ▼
  VersedRetriever._get_relevant_documents()
     │  top-k cosine-distance SQL query against the existing `chunk` table
     │  NO version filter — retrieves across every ingested version equally
     ▼
  format_docs (chunk content + version + source path, for citation-shaped output)
     │
     ▼
  LCEL chain: retriever | format_docs | prompt | ChatOllama(phi3.5-mini) | StrOutputParser
     │
     ▼
  print answer to stdout
```

**`VersedRetriever`** (`src/versed/retrieval.py`) subclasses
`langchain_core.retrievers.BaseRetriever`. It wraps the existing `chunk`
table's SQLAlchemy session rather than introducing LangChain's own
`PGVector` vector store — the Foundation plan's schema stays the single
source of truth for chunks, and this retriever is the seam Milestone 5's
hybrid retrieval will extend (add BM25, add a version filter, add
reranking) rather than throwaway scaffolding.

No version filter is a deliberate design choice, not an oversight: this is
what makes the baseline capable of confidently returning stale answers,
which is the exact behavior the golden set's removed-API and migration
categories are built to expose, and the exact gap Milestone 5 closes.

**`src/versed/rag.py`** builds the LCEL chain and the prompt template. The
prompt states the retrieved chunks as untrusted context and asks for an
answer grounded in them — no citation *enforcement* yet (that requires the
schema/verification machinery from spec section 6.6, Milestone 7), just
citation-shaped instructions so the prompt format doesn't need to change
later.

CLI-only interface, consistent with the Foundation plan's `versed.cli`
pattern (`ingest-docs`, `build-symbols`, `timeline`). No FastAPI in this
plan — that belongs to Milestone 8's "production shape."

## 5. Golden set generation

**Category breakdown (150 total, per parent spec section 8):**

| Category | n | Generation source |
|---|---|---|
| Version-explicit | 40 | `symbol_event` rows (`removed`/`deprecated`/`moved`), question names an explicit `from_version` |
| Version-implicit | 30 | current `symbol` rows, question omits version, answer is that symbol's current recorded signature/docstring |
| Migration | 30 | `symbol_event` ranges (`from_version` → `to_version`), answer pulled from the event's `detail`/`alternative` fields |
| Removed API | 30 | `symbol_event` rows where `event_type = removed`, answer names the recorded replacement |
| Unanswerable | 20 | symbols/topics verified absent from the corpus entirely (nonexistent name, or out-of-scope topic), `should_abstain = true`, no `expected_answer` |

For version-implicit rows, the question text omits a version, but the
generated `eval_question.target_version` field is still recorded as the
actual current version resolved at generation time — "current" is a fact
that must be pinned at generation time, not left implicit in storage, or
grading against it later (Milestone 6) would have no fixed target. For the
unanswerable category, "verified absent" means the generator checks the
candidate symbol name against the `symbol` table across every ingested
version (confirming it never existed) or picks a topic outside LangChain's
domain entirely — not a guess.

Every category is **fully mechanically drafted** — no category requires the
reviewer to supply an answer from memory. Each drafted row carries the exact
source record it was derived from (the `symbol_event` or `symbol` row, or
"verified absent" for the unanswerable category) so review means checking a
claim against its cited evidence, not authoring from expertise. This
addresses a real constraint: no LangChain subject-matter expert is available
on this project, so ground truth cannot depend on a reviewer's memorized
knowledge — only on data already proven correct by the Foundation plan's
introspection pipeline.

**File format:** YAML (`data/golden_set.yaml`), not JSON — long-form
question/answer text is markedly easier to hand-edit and diff-review in
YAML, and this file exists specifically to be hand-edited in a PR.

**Workflow:**

```
versed golden-set generate     # writes data/golden_set.yaml with all 150 drafted rows
  (you hand-review/edit the file directly, as a normal PR diff)
versed golden-set load         # validates + upserts the reviewed file into eval_question
```

`generate` is idempotent and safe to re-run (e.g., after a new version is
ingested) — it regenerates drafts, it does not merge with prior hand-edits.
Re-running after review has started would discard edits, so it's a
start-of-review action, not a repeatable sync; this is acceptable because
golden-set v1 is a one-time construction, not an ongoing pipeline.

## 6. Data model additions

`eval_question` table, exactly as specified in the parent spec section 7:

```sql
eval_question(id, question, category, target_version, expected_answer,
              expected_symbols text[], should_abstain, human_label)
```

New Alembic migration adds this table. The YAML file carries two
reviewer-facing fields per entry beyond the `eval_question` columns:
`reviewed` (bool, gates whether `load` will accept the row at all — any
entry missing `reviewed: true` is rejected, so an unreviewed draft can't
silently become ground truth) and an optional freeform `reviewer_note`
(str), which `load` copies verbatim into `eval_question.human_label` if
present, else leaves that column null.

## 7. Error handling

- **Empty retrieval** (no chunks match, or table is empty): the chain still
  calls the LLM with no context. This is the naive baseline's documented
  failure mode, not a condition to guard against — guarding it away would
  hide exactly the behavior Milestone 5 is meant to fix.
- **Ollama unreachable**: the `langchain-ollama`/`httpx` error surfaces
  as-is. No retry wrapper here, consistent with the Foundation plan's
  existing `ponytail:`-marked deferrals of retry logic elsewhere in the
  pipeline.
- **`golden-set load` on a malformed or incompletely-reviewed YAML file**:
  fails fast via pydantic validation before writing anything to the
  database — no partial loads.

## 8. Testing

- `VersedRetriever`: integration test against the real test Postgres
  (existing `integration` marker), no LLM involved — asserts top-k ordering
  and that results span multiple versions (proving the "no filter" property
  holds).
- LCEL chain: one `slow`-marked end-to-end test that calls local Ollama for
  real, plus one fixture-recorded test with a canned response for fast/
  deterministic CI runs.
- `generate_golden_set()`: unit test asserting exact per-category counts
  (40/30/30/30/20) and that every generated row's `expected_answer` matches
  its cited source record verbatim (province of correctness-by-construction,
  not judgment).
- `data/golden_set.yaml` itself: a committed-file validation test — the file
  as it exists in the repo must parse against the `EvalQuestion` pydantic
  model and every row must have `reviewed: true` before this plan's tasks
  are considered done. This is what makes "hand-reviewed" enforceable rather
  than aspirational.

## 9. Risks

- **Phi-3.5-mini output quality.** It's a small model; baseline answers may
  be noticeably rough. This is acceptable and expected — Milestone 3's job
  is to be a beatable floor, not a good product.
- **Mechanical generation coverage.** Some conceptual question phrasings
  the symbol table genuinely cannot express (the parent spec's original
  concern) are traded away in favor of 100% reviewer feasibility without an
  SME. If this later proves to leave real gaps in what the golden set can
  test, RAGAS-assisted *question* drafting (not just metrics) becomes a
  candidate to revisit for a golden-set v2 — not for this plan.
