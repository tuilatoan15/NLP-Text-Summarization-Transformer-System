# De Tai 18 - AI Document Intelligence Platform

He thong duoc nang tu web summarize demo thanh nen tang nghien cuu Document Intelligence, tap trung vao so sanh Extractive Summarization va Abstractive Summarization cho tieng Viet.

## Research Goals

1. Danh gia extractive vs abstractive bang cung mot pipeline ingest.
2. Giu citation grounding cho tung cau summary.
3. Ho tro long-document understanding bang semantic chunking va summary tree.
4. Truc quan hoa metrics, chunks, embeddings va ranking.
5. Sinh artefact kieu NotebookLM: overview, quiz, flashcards, report, podcast script, mindmap, presentation.

## Architecture

```text
frontend/
  React + Tailwind + Recharts + Framer Motion
  pages/DocumentIntelligence.jsx

api/
  main.py
  document_intelligence.py

src/
  document_intelligence.py
  dashboard.py
  extractive.py
  abstractive.py
  evaluate.py
  fact_check.py

loaders/ preprocess/ embeddings/ pipeline/
  production ingest subsystem

storage/document_intelligence/
  local document state, analysis outputs, visualization payloads

docs/
  DATABASE_SCHEMA.sql
```

## Backend Flow

1. `POST /documents/ingest`
   Upload PDF/DOCX/TXT, extract text, preserve headings/tables/bullets, clean Vietnamese text, chunk semantically, embed chunks, generate overview/assets/visualization.

2. `POST /documents/{document_id}/search`
   Semantic retrieval over chunks. If a heavy embedding model is unavailable, the system falls back to deterministic hash embeddings or lexical overlap so offline demos still work.

3. `POST /documents/{document_id}/compare`
   Runs TextRank, LexRank, LSA, ViT5, BARTPho depending on configuration. Every summary receives citations, consistency checks and a research matrix.

4. `GET /documents/{document_id}/assets`
   Returns overview, research report, quiz, flashcards, mindmap, presentation, infographic, timeline and entity graph.

5. `GET /documents/{document_id}/visualization`
   Returns chunk graph, embedding PCA map, similarity heatmap and hierarchy tree.

## Research Design

Extractive algorithms:

- TextRank: graph ranking sentence importance.
- LexRank: thresholded similarity graph.
- LSA: latent semantic concept scoring.
- TF-IDF ranking: planned additional baseline.

Abstractive algorithms:

- ViT5.
- mT5.
- BARTPho.
- Recursive and hierarchical summarization for long documents.

Evaluation:

- ROUGE-1/2/L.
- BLEU.
- BERTScore.
- Semantic similarity.
- Compression ratio.
- Latency.
- Factual consistency.
- Citation grounding coverage.

## Hallucination Mitigation

- Summary sentence to top source chunks.
- `supported / suspicious / unsupported` consistency status.
- Extractive summaries are treated as factual anchors.
- Abstractive outputs are audited against retrieved source evidence.
- Long-document summarization uses chunk summaries and section summaries before global synthesis.

## Production Notes

- PostgreSQL stores document metadata, chunks, summaries, evaluations and generated artefacts.
- Redis stores job state, cache keys and websocket progress.
- MinIO stores original files and generated audio/report exports.
- ChromaDB or FAISS stores vectors.
- Local JSON storage remains the development fallback for a lightweight thesis demo.

## Low VRAM Settings

- `PRELOAD_MODELS=0` for lazy loading.
- `MAX_GPU_CONCURRENT=1`.
- Embedding `batch_size=2..4`.
- `use_fp16=true`.
- Use `embedding_model=hash` during UI demos.
- Run heavy embedding benchmarks one model at a time.

## Implemented Surface

- `/documents/ingest`
- `/documents/{id}/search`
- `/documents/{id}/compare`
- `/documents/{id}/assets`
- `/documents/{id}/visualization`
- `/documents/{id}/stream`
- Frontend `/documents`

## Next Research Extensions

- Add TF-IDF ranking endpoint into the extractive registry.
- Persist records into PostgreSQL instead of local JSON.
- Add FAISS/Chroma vector store writer.
- Add NLI contradiction detector for stronger hallucination checks.
- Add TTS export for podcast MP3.
- Add UMAP/t-SNE optional projection for richer embedding visualization.
