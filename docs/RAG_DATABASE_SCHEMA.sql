-- RAG chat persistence schema (implemented in backend/services/rag/repository.py).
-- This logical schema mirrors SQLite storage and can be adapted to Postgres.

CREATE TABLE rag_documents (
  id TEXT PRIMARY KEY,
  filename TEXT NOT NULL,
  source_type TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  metadata_json TEXT NOT NULL
);

CREATE TABLE rag_chunks (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  page INTEGER,
  chunk_index INTEGER NOT NULL,
  text_content TEXT NOT NULL,
  embedding_model TEXT NOT NULL,
  metadata_json TEXT NOT NULL
);

CREATE TABLE rag_embeddings (
  chunk_id TEXT PRIMARY KEY,
  vector_json TEXT NOT NULL,
  dimension INTEGER NOT NULL,
  model_name TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE rag_conversations (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE rag_messages (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  confidence REAL,
  retrieval_threshold REAL,
  citations_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

