# Versed — Version-Aware Documentation QA

**Status:** Design
**Date:** 2026-09-02
**Author:** avishmaniar24@gmail.com

---

## 1. Problem

Ask any documentation chatbot "how do I build an agent in LangChain" and it
answers with `create_react_agent` from `langgraph.prebuilt` — an API that moved
and was renamed in v1.0. The answer is fluent, cited, and wrong.

This happens because naive RAG indexes documentation from many releases into one
flat vector store. Chunks from v0.1 and v1.3 sit side by side with nothing to
distinguish them. Retrieval scores them on semantic similarity alone, so the
model receives contradictory context and picks whichever chunk embedded best.
The user gets a confident answer describing an API that no longer exists.

Versioned documentation is not a niche case. It is the default state of every
actively maintained library, framework, and internal platform. Any organization
running RAG over its own engineering docs has this problem and usually does not
know it, because standard RAG evaluation measures faithfulness to retrieved
context — and the system *is* faithful to context that happens to be four
releases stale.

## 2. Goals

1. Answer documentation questions correctly for a specified or inferred target
   version.
2. Refuse or warn — rather than fabricate — when the asked-about API does not
   exist in the target version.
3. Answer migration questions ("how do I move this from 0.3 to 1.0") using
   structured knowledge of what actually changed.
4. Measure all of the above against a human-verified golden set, with the
   evaluation gating CI.
5. Withstand indirect prompt injection delivered through the retrieved
   documentation itself.
6. Run as a deployed, observable, load-tested HTTP service.

## 3. Non-goals

- Not a general-purpose chatbot. Out-of-scope questions are refused, and that
  refusal is a measured behavior, not a fallback.
- Not multi-library. One corpus, deeply handled, beats five handled shallowly.
- Not a fine-tuning project. Retrieval and orchestration only.
- No user accounts, no chat history persistence beyond a session, no billing.
  These add build time without adding a single interview-relevant signal.

## 4. Success criteria

The project succeeds when the README opens with a table like this, populated
with real measured numbers:

| Metric | Naive RAG baseline | Versed | Notes |
|---|---|---|---|
| Stale-answer rate | high | low | headline metric |
| Answer correctness | baseline | improved | LLM judge, human-calibrated |
| Retrieval recall@10 | baseline | improved | version-filtered vs not |
| Faithfulness | baseline | improved | claims grounded in cited spans |
| Abstention F1 | near zero | high | knows when to refuse |
| Injection attack success | high | low | before/after guardrail layers |
| Judge–human agreement (κ) | — | reported | credibility of every row above |
| p95 latency @ 20 RPS | — | reported | |
| Cost per 1k queries | — | reported | |

Targets are deliberately not fixed in advance. Inventing a number to hit invites
tuning the eval to the target. The baseline is measured first, and whatever
delta the system honestly produces is what gets reported — including places it
fails to improve, which are more interesting to discuss in an interview than a
clean sweep.

## 5. Corpus

**Library:** LangChain (Python), plus LangGraph where the two interact.

**Versions:** 0.1, 0.2, 0.3, 1.0, and current 1.x — five points spanning the
library's most disruptive period.

Known breaking changes across that range, each a natural golden-set question:

| Change | From | To |
|---|---|---|
| `LLMChain`, `RetrievalQA`, `ConversationalRetrievalChain` removed | 0.2 | 0.3 |
| Pydantic 1 → 2 internally; Pydantic 1 support dropped | 0.2 | 0.3 |
| `langgraph.prebuilt.create_react_agent` → `langchain.agents.create_agent` | 0.3 | 1.0 |
| `prompt` parameter renamed `system_prompt` | 0.3 | 1.0 |
| `langchain.retrievers` → `langchain_classic.retrievers` | 0.3 | 1.0 |
| `.text()` method → `.text` property | 0.3 | 1.0 |
| Minimum Python 3.9 → 3.10 | 0.3 | 1.0 |

**Acquisition strategy — two independent sources, deliberately:**

*Prose corpus* comes from git, not the live docs site. LangChain's documentation
moved from `python.langchain.com` to `docs.langchain.com` mid-history, and old
versions are archived at inconsistent URLs. Scraping is fragile and
irreproducible. Instead, check out the docs directory at each release tag from
the `langchain-ai/langchain` and `langchain-ai/docs` repositories. Reproducible,
diffable, offline, and no rate limits.

*Symbol timeline* comes from the packages themselves. For each target version,
create an isolated venv, `pip install langchain==<version>`, and walk the module
tree to record every public symbol, its signature, and its docstring. LangChain
decorates deprecated APIs with `@deprecated(since=, removal=, alternative=)`,
so deprecation metadata is extracted programmatically rather than inferred by an
LLM.

This second source is the project's core asset. It produces ground truth about
what existed when, which lets the system answer version questions
deterministically instead of hoping the language model notices a version string
in a retrieved chunk. It also generates a large portion of the golden set
automatically and correctly.

## 6. Architecture

```
                    ┌──────────────────────────────────┐
   ingest (offline) │  docs @ git tags   packages @ pip │
                    └────────┬──────────────┬──────────┘
                             │              │
                    ┌────────▼───────┐  ┌───▼──────────────┐
                    │ chunker        │  │ symbol extractor │
                    │ (structural)   │  │ (AST + import)   │
                    └────────┬───────┘  └───┬──────────────┘
                             │              │
                    ┌────────▼──────────────▼──────────┐
                    │ Postgres: chunks + pgvector       │
                    │           symbol_timeline         │
                    └────────┬──────────────────────────┘
                             │
   serve (online)   ┌────────▼──────────────────────────┐
                    │ FastAPI  →  LangGraph answer graph │
                    │            ├─ retriever (hybrid)   │
                    │            ├─ timeline lookup      │
                    │            └─ guardrail layers     │
                    └────────┬──────────────────────────┘
                             │
                    ┌────────▼───────┐   ┌───────────────┐
                    │ Redis cache    │   │ eval harness  │
                    └────────────────┘   │ (offline, CI) │
                                         └───────────────┘
```

Seven units, each independently testable:

### 6.1 `ingest.docs` — prose corpus builder
**Does:** checks out docs at each release tag, walks markdown/MDX, emits chunks.
**Interface:** `build_corpus(versions: list[str]) -> Iterator[Chunk]`
**Depends on:** git, filesystem. No database, no network at call time.

Chunking is structural, not fixed-width. Splits on heading boundaries, never
inside a fenced code block, and carries the full heading path as context. Each
chunk records: version, source path, heading path, symbols mentioned, and
whether it contains executable code.

### 6.2 `ingest.symbols` — symbol timeline builder
**Does:** installs each version in an isolated venv, introspects the module
tree, extracts signatures, docstrings, and `@deprecated` metadata.
**Interface:** `build_timeline(versions: list[str]) -> Iterator[SymbolRecord]`
**Depends on:** `venv`, `pip`, `importlib`, `inspect`, `ast`.

Introspection runs in a subprocess per version, since the versions cannot
coexist in one interpreter. Output is a per-version symbol table; the timeline
is then derived by diffing consecutive versions to classify each symbol as
`added`, `changed`, `deprecated`, `moved`, or `removed`.

### 6.3 `store` — persistence
**Does:** owns the schema and all queries. Nothing else touches SQL.
**Interface:** `upsert_chunks()`, `upsert_symbols()`, `search_hybrid()`,
`lookup_symbol()`, `symbol_history()`
**Depends on:** Postgres + pgvector, Alembic migrations.

### 6.4 `retrieval` — hybrid retriever
**Does:** BM25 + dense search, version filtering, reranking, parent expansion.
**Interface:** `retrieve(query: str, version: str, config: RetrievalConfig) -> list[Chunk]`
**Depends on:** `store`, an embedding model, a cross-encoder reranker.

Every stage is switchable through `RetrievalConfig` — this is what makes the
ablation matrix a config sweep rather than six branches of code.

### 6.5 `graph` — the LangGraph answer pipeline
**Does:** orchestrates one question into one answer.
**Interface:** `answer(question, version_hint, session_id) -> AsyncIterator[Event]`
**Depends on:** `retrieval`, `store`, `guardrails`, an LLM.

```
query_analysis        extract intent, target symbols, explicit version
   ↓
version_resolution    explicit hint → inferred from query → default latest
   ↓
timeline_check        does this symbol exist in target version?
   ↓                  ├─ removed/moved → migration branch
retrieve              hybrid, version-filtered
   ↓
grade_documents       are these actually relevant?
   ↓                  ├─ no → relax filter, retrieve once more
conflict_detection    do retrieved chunks disagree across versions?
   ↓
generate              answer with version-tagged citations
   ↓
self_check            is every claim grounded in a cited span?
   ↓                  ├─ no → regenerate once, then abstain
answer / abstain
```

State is checkpointed so a failed run is resumable and inspectable. All edges
out of `grade_documents` and `self_check` are conditional, with a hard cap on
retries so no query can loop.

### 6.6 `guardrails` — input, tool, and output defenses
**Does:** enforces that retrieved documentation is treated as data, never as
instructions.
**Interface:** `scan_input()`, `wrap_context()`, `verify_citations()`, `scan_output()`
**Depends on:** `store` (to verify cited spans exist verbatim).

Layers, each independently measurable:
1. Instruction/content separation — retrieved text is delimited and the system
   prompt states it is untrusted data.
2. Structured output — the model returns a schema with claims and citation IDs,
   not free prose.
3. Citation verification — every claim's cited span must exist verbatim in a
   retrieved chunk. Unverifiable claims are dropped.
4. Output scanning — secrets, URLs to non-documentation domains, PII.
5. Scope refusal — questions unrelated to the corpus are declined.

### 6.7 `evals` — the measurement harness
**Does:** runs the golden set against any config, scores it, compares to a
baseline, fails CI on regression.
**Interface:** `run_suite(config) -> SuiteResult`, `compare(a, b) -> Diff`
**Depends on:** `graph`, a judge LLM, the golden set on disk.

Detailed in section 8. This unit is extracted into a standalone package in
milestone 7.

## 7. Data model

```sql
chunk(id, version, source_path, heading_path, content, embedding vector,
      has_code, symbols_mentioned text[], token_count)

symbol(id, version, qualified_name, kind, signature, docstring,
       deprecated_since, removal_version, alternative)

symbol_event(qualified_name, from_version, to_version, event_type, detail)
      -- event_type: added | changed | deprecated | moved | removed

eval_question(id, question, category, target_version, expected_answer,
              expected_symbols text[], should_abstain, human_label)

eval_run(id, config_hash, git_sha, started_at, metrics jsonb)
eval_result(run_id, question_id, answer, retrieved_ids, scores jsonb)
```

`symbol_event` is the table that makes migration answers possible: a query about
moving from 0.3 to 1.0 becomes a range scan over events, not a retrieval problem.

## 8. Evaluation design

**Golden set: 150 questions across five categories.**

| Category | ~n | Tests | Expected behavior |
|---|---|---|---|
| Version-explicit | 40 | version filtering | correct answer for stated version |
| Version-implicit | 30 | sane defaults | answers for current, states which |
| Migration | 30 | cross-version reasoning | describes the actual change |
| Removed API | 30 | staleness resistance | warns; names replacement |
| Unanswerable | 20 | abstention | refuses without fabricating |

Roughly half the questions in the version-explicit, migration, and removed-API
categories are generated from `symbol_event` — the ground truth is already
known, so they are correct by construction. The remainder are hand-written to
cover conceptual questions the symbol table cannot express. All 150 are reviewed
by hand before entering the set; generated does not mean unverified.

**Metrics.**

*Retrieval:* recall@k, MRR, and version-precision (fraction of retrieved chunks
belonging to the target version).

*Generation:* answer correctness, faithfulness, citation accuracy (do cited
spans exist and support the claim), and **stale-answer rate** — the fraction of
answers referencing a symbol absent from the target version. Stale-answer rate
is computed against `symbol_event`, so it is exact and needs no LLM judge.
It is the headline metric precisely because it cannot be argued with.

*Behavioral:* abstention precision and recall.

*Operational:* p50/p95 latency, tokens and cost per query.

**Judge calibration.** The correctness and faithfulness judges are LLM-based, so
their agreement with human judgment is itself measured: 50 responses are labeled
by hand, Cohen's κ is computed against the judge, and κ is reported alongside
every metric that depends on it. If agreement is poor the judge prompt is
revised until it is acceptable — and both the before and after are reported,
because the process is more interesting than the final number.

**Ablation matrix.** Chunking (fixed / structural) × retrieval (BM25 / dense /
hybrid) × rerank (on / off) × version filter (on / off), scored on correctness,
stale-answer rate, latency, and cost. The naive baseline — fixed chunking, dense
only, no rerank, no version filter — is the "typical tutorial RAG" configuration
and is what the README compares against.

**CI gate.** A GitHub Action runs a 40-question subset on every PR and fails it
when correctness or faithfulness drops more than 2 points against the committed
baseline. The full 150 run nightly. At least one PR that the gate legitimately
caught is preserved and linked from the README.

**Determinism.** LLM calls in the test suite are recorded and replayed from
fixtures, so unit and integration tests are free, fast, and deterministic. Only
eval runs hit live APIs.

## 9. Adversarial testing

Attack corpus: roughly 100 injection payloads planted inside otherwise-normal
documentation chunks. Categories: direct instruction override, fake system
messages, data-exfiltration requests, citation forgery, and refusal bypass.

Procedure: measure attack success rate with all guardrails off, then enable each
layer from section 6.6 in turn and re-measure. The result is a table showing
each layer's marginal contribution — including any layer that turns out not to
help, which is worth reporting honestly.

Attack success is defined mechanically: the response contains the injected
target string, calls a disallowed action, or cites a span that does not exist.
No judge required.

## 10. Service and operations

**API:** FastAPI. `POST /v1/ask` (SSE streaming), `POST /v1/compare` (returns
both naive-baseline and Versed answers for the demo), `GET /v1/symbols/{name}`
(timeline lookup), `GET /health`, `GET /metrics`.

**Storage:** Postgres 16 + pgvector, Alembic migrations. Redis for semantic
caching of embeddings and full answers; cache hit rate and the cost it saves are
measured and reported.

**Testing:** pytest. Unit tests per module; integration tests against a
containerized Postgres; recorded-fixture tests for anything touching an LLM.

**CI:** GitHub Actions — lint, type-check, test, then the eval gate.

**Observability:** structured JSON logging with request IDs, OpenTelemetry
tracing spanning the LangGraph nodes, LangSmith for LLM-specific traces,
Prometheus metrics at `/metrics`.

**Deployment:** Docker Compose for local, a single deployed instance with a
public URL for the demo. Load test with Locust reporting p50/p95 at 20 RPS.

**Demo UI:** one page, side-by-side. Same question, naive RAG on the left,
Versed on the right, citations and version badges visible. This page is what a
recruiter actually looks at, so it gets real attention despite being small.

## 11. Extracted library

After the service is complete, the `evals` unit is extracted into
`versed-eval`, a standalone package: a pytest plugin plus CLI for evaluating RAG
systems, with judge calibration and CI gating built in. Typed public API,
documentation, semantic versioning, published to PyPI.

The extraction is not cosmetic. Pulling it out forces the harness to stop
depending on this project's internals, which is a genuine design exercise and a
better interview story than the harness alone.

## 12. Milestones

Each milestone ends in a working, committed, demonstrable state.

1. **Corpus** — docs checked out at 5 tags, structurally chunked, in Postgres.
2. **Symbol timeline** — per-version introspection, diffed into `symbol_event`,
   queryable. *Deliverable: the timeline for `create_react_agent` printed from
   the CLI.*
3. **Baseline RAG** — naive config end to end, so there is something to beat.
4. **Golden set v1** — 150 questions, hand-reviewed, in the repo.
5. **The real system** — hybrid version-filtered retrieval, the LangGraph
   pipeline, citations, abstention.
6. **Evaluation** — full metric suite, judge calibration, ablation matrix, CI
   gate. *Deliverable: the README results table.*
7. **Guardrails** — attack corpus, layered defenses, before/after table.
8. **Production shape** — Redis, tracing, metrics, load test, deployment, demo
   UI.
9. **Library** — extract `versed-eval`, publish.
10. **Publication** — writeup, datasets on HuggingFace, README polish.

Milestones 1–6 constitute a resume-ready project on their own. 7–10 deepen it.

## 13. Risks

**Docs restructuring across versions.** LangChain's documentation moved domains
and reorganized directories mid-history, so file paths do not align across tags.
*Mitigation:* the symbol timeline is the authoritative version signal and is
independent of docs layout. Prose chunks are matched to symbols by name, not by
path. If a given version's docs prove unusable, that version is dropped from the
prose corpus while remaining in the timeline.

**Old versions uninstallable.** `pip install langchain==0.1.x` may fail on
modern Python. *Mitigation:* pin an appropriate interpreter per version via
`uv`, which can fetch older Pythons. If a version genuinely cannot be installed,
fall back to AST parsing of the source tarball — no execution required.

**Golden set bias.** Questions generated from `symbol_event` and answered by a
system that consults `symbol_event` risks measuring the pipeline against itself.
*Mitigation:* the hand-written portion of each category is scored separately and
reported separately. If the two diverge sharply, that is a finding to report,
not a bug to hide.

**Judge unreliability.** *Mitigation:* κ is measured and published; the headline
metric (stale-answer rate) is computed mechanically and does not involve a judge
at all.

**Scope creep.** The design already spans seven units. *Mitigation:* the
non-goals in section 3 are binding. New ideas go in a `FUTURE.md`, not in the
build.

## 14. What this demonstrates

Mapped deliberately, since the project exists to be discussed in interviews:

| Skill | Where it shows up |
|---|---|
| LangChain | ingestion, retrievers, LCEL composition |
| LangGraph | the answer graph — conditional edges, checkpointing, streaming |
| Advanced RAG | hybrid search, reranking, metadata filtering, parent expansion |
| Evaluation | golden set, judge calibration, ablations, CI gating |
| Guardrails | indirect injection defense with measured before/after |
| Backend engineering | FastAPI, Postgres, Redis, Docker, migrations, load testing |
| Testing discipline | deterministic LLM tests via recorded fixtures |
| Data engineering | version introspection, diffing, derived timeline tables |
| Ops | tracing, metrics, deployment, cost accounting |
