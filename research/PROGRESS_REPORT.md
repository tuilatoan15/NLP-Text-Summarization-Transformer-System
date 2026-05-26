# Progress Report — Iteration 2 (2026-05-26)

This report tracks progress toward the full “AI Document Intelligence” specification (I–XXII). Percentages are deliberately conservative and reflect “works end-to-end locally” rather than “scaffold exists”.

## Overall completion

- **Overall completion (end of Iteration 2): 60%**
- Prior baseline after Iteration 1 (estimated): **~45%**
- Net gain this iteration: **+15%**

## I–XXII checklist mapping (per-section %)

### I. Goals (upload, compare, eval, viz, Vietnamese, explainability, retrieval, citation, long-doc)
- **75%**
- Done this iteration:
  - Added **explainability** ranking graph (nodes/edges) and evidence highlights.
  - Added **hierarchical (map-reduce) summarization endpoint** for long docs.
  - Added **semantic+lexical hallucination audit** in compare outputs.
- Remaining:
  - True NotebookLM-style multi-turn workspace and deeper semantic reasoning.

### II. Architecture (React+TS, Tailwind, Recharts, Motion, FastAPI, async, WS, transformers, FAISS, Chroma, Postgres, Redis, MinIO)
- **60%**
- Done this iteration:
  - Introduced **incremental TypeScript** in frontend (tsconfig + typed API client + TS pages).
  - Added async API flow in `/documents` ingest/compare.
  - Postgres repository improved and aligned with schema (import-safe).
- Remaining:
  - Full frontend TS migration, backend async wiring (avoid `asyncio.run` in sync paths), robust production job queue.

### III. Main features (1–10)
- **65%**
- Done this iteration:
  - Explainability API + UI route.
  - Hierarchical summarization API.
  - Improved hallucination detection payload.
- Remaining:
  - Stronger semantic search UX with chunk highlighting; citation viewer with evidence excerpts per sentence across all models.

### IV. NotebookLM features (1–10)
- **45%**
- Done this iteration:
  - Added **podcast TTS export endpoint** (gTTS, graceful text-only fallback).
- Remaining:
  - Better generation quality (LLM-backed), export formats (PDF/Docx), mindmap rendering, presentation export.

### V. Ingestion pipeline
- **70%**
- Done this iteration: no major changes (stable).
- Remaining: stronger OCR fallback evaluation and structure fidelity on complex PDFs.

### VI. Chunking system
- **65%**
- Done this iteration: no major changes (stable).
- Remaining: section-aware chunk tree persisted to DB; better token-aware truncation.

### VII. Embedding benchmark
- **70%**
- Done this iteration:
  - Added `scripts/benchmark_embeddings.py` to compare **hash vs sentence-transformer** models quickly.
- Remaining:
  - Produce standardized benchmark reports in `research/` and add CI artifact export.

### VIII. Extractive + explainability
- **80%**
- Done this iteration:
  - `build_sentence_ranking_graph()` added to `src/explainability.py`.
  - `/documents/{id}/explainability` endpoint.
  - UI screen `/documents/explainability` listing ranked sentences.
- Remaining:
  - Visual graph rendering (force/graph view) instead of list + edges count.

### IX. Abstractive + hierarchical
- **70%**
- Done this iteration:
  - `summarizers/hierarchical.py` map-reduce flow.
  - `/documents/{id}/summarize/hierarchical`.
- Remaining:
  - Better reduce prompts/strategies, chunk selection by retrieval, controllable length per section.

### X. Evaluation metrics
- **65%**
- Done this iteration:
  - Added memory snapshot + readability metrics into `src.evaluate.evaluate_summary`.
  - Evaluation UI chart for latency/memory.
- Remaining:
  - ROUGE/BERTScore batch evaluation reports and richer dashboards.

### XI. Hallucination detection
- **60%**
- Done this iteration:
  - `evaluation/hallucination.py` now includes **sentence-level semantic alignment** (SentenceTransformer if available, lexical fallback).
  - Compare response now uses `audit_summary()` as “consistency” payload.
- Remaining:
  - Add a lightweight NLI contradiction model (optional), plus source verification using retrieved evidence chunks.

### XII. Frontend screens (1–10)
- **60%**
- Done this iteration:
  - Routes split under `/documents/*`: upload, analysis, compare, evaluation, search, explainability, notebook.
  - New TS pages for those routes.
- Remaining:
  - Dashboard-level document list + persisted history, citation viewer improvements, mindmap rendering, semantic search highlighting.

### XIII. Backend modular
- **70%**
- Done this iteration:
  - `DocumentService` now offers async ingest/compare used by API.
  - Postgres repository aligned with schema and import-safe.
- Remaining:
  - Remove `asyncio.run()` sync calls; unify to fully async and add background jobs.

### XIV. Project structure
- **75%**
- Done this iteration:
  - Added hierarchical summarization module, TTS service, benchmark script.
- Remaining:
  - Clean up deprecated/duplicate modules; align fully to target folder plan (backend/, frontend/, ai_models/, embeddings/, summarizers/, evaluation/, visualization/, datasets/, research/, scripts/, docs/).

### XV. Database tables
- **55%**
- Done this iteration:
  - Repository writes for: documents, chunks, summaries, evaluations, research_runs, podcasts.
- Remaining:
  - Wire embeddings table (vector_ref + backend), generated_reports/quizzes persistence, proper migrations.

### XVI. Visualizations
- **65%**
- Done this iteration:
  - Evaluation tab now charts latency/memory.
- Remaining:
  - True graph visualization for explainability; UMAP optional projection; chunk hierarchy tree view.

### XVII. Dataset pipeline
- **70%**
- Done this iteration: no major changes.
- Remaining: richer dataset statistics exports, automatic cleaning audits.

### XVIII. Performance (fp16, batching, cache, stream)
- **55%**
- Done this iteration:
  - Added system metrics snapshot returned in compare response.
- Remaining:
  - Streaming of long hierarchical summarization, background queue, caching for embeddings and summaries.

### XIX. Research report helpers
- **60%**
- Done this iteration:
  - Bench script emits JSON to `research/embedding_benchmark.json`.
- Remaining:
  - Automated evaluation report export in `research/` and notebooks.

### XX. Code quality
- **60%**
- Done this iteration:
  - Added tests for explainability, hierarchical summarization, hallucination audit.
- Remaining:
  - Linting/formatting, type checks in CI for frontend.

### XXI. README
- **55%**
- Done this iteration:
  - Run instructions unchanged; README update not strictly required.
- Remaining:
  - Document new endpoints and TypeScript route split under `/documents/*`.

### XXII. Docker + scripts
- **60%**
- Done this iteration:
  - Added benchmark script; CI workflow added for Python tests.
- Remaining:
  - Frontend build/test in CI; docker-compose env flags for DB persistence and TTS.

## What was done in Iteration 2 (high impact)

1. **Postgres persistence (real wiring)**: repository now aligns with `docs/DATABASE_SCHEMA.sql` and is import-safe; `DocumentService` persists ingest + compare results when enabled.
2. **Routes + TypeScript (incremental)**: `/documents/*` split into Upload/Analysis/Compare/Evaluation/Search/Explainability/Notebook, with TS API client + TS pages.
3. **Explainability + long-doc**: new explainability API (`ranking_graph`) and hierarchical summarization endpoint.

## What remains for Iteration 3 (top priorities)

1. **Finish Postgres persistence**: embeddings table, generated_reports/quizzes/podcasts metadata, and avoid `asyncio.run()` by making routers fully async end-to-end.
2. **Frontend TS migration**: convert remaining shared components/layouts to TS, add stronger types for API payloads, and improve visualization components.
3. **Hallucination & citations**: evidence excerpts per summary sentence (top chunks), plus optional lightweight NLI contradiction classifier.

## How to verify (Iteration 2 additions)

### Backend endpoints

1. Ingest:
```bash
curl -F "file=@./some.txt" -F "include_embeddings=true" -F "embedding_model=hash" http://localhost:8000/documents/ingest
```

2. Explainability graph:
```bash
curl "http://localhost:8000/documents/{document_id}/explainability?algorithm=textrank"
```

3. Hierarchical summarization:
```bash
curl -X POST http://localhost:8000/documents/{document_id}/summarize/hierarchical \\
  -H "Content-Type: application/json" \\
  -d "{\"model_key\":\"vit5\",\"use_extractive_map\":true}"
```

4. Podcast TTS export:
```bash
curl -X POST http://localhost:8000/documents/{document_id}/podcast/tts
```

### Frontend verification

Run frontend and navigate to:

- `http://localhost:5173/documents/upload`
- `http://localhost:5173/documents/explainability`
- `http://localhost:5173/documents/evaluation`

### Postgres persistence verification

1. Ensure docker-compose Postgres is running.
2. Set env:
```bash
ENABLE_DB_PERSISTENCE=1
```
3. Ingest + compare, then query tables:
```sql
SELECT COUNT(*) FROM uploaded_documents;
SELECT COUNT(*) FROM chunks;
SELECT COUNT(*) FROM summaries;
SELECT COUNT(*) FROM evaluations;
```

