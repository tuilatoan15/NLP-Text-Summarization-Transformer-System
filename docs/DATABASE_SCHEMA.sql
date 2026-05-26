-- Research-grade schema for Agentic AI Document Intelligence.
-- PostgreSQL 15+ recommended. Embeddings can be stored in Chroma/FAISS/MinIO;
-- this schema stores metadata and optional vector references.

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    display_name TEXT,
    role TEXT NOT NULL DEFAULT 'researcher',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS uploaded_documents (
    id TEXT PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    title TEXT,
    source_filename TEXT NOT NULL,
    source_type TEXT NOT NULL,
    storage_uri TEXT,
    language TEXT,
    page_count INTEGER,
    word_count INTEGER,
    quality_score DOUBLE PRECISION,
    metadata JSONB NOT NULL DEFAULT '{}',
    structure JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES uploaded_documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    section_path JSONB NOT NULL DEFAULT '[]',
    page_start INTEGER,
    page_end INTEGER,
    token_count INTEGER,
    word_count INTEGER,
    keywords JSONB NOT NULL DEFAULT '[]',
    semantic_tags JSONB NOT NULL DEFAULT '[]',
    parent_section TEXT,
    overlap_from_previous BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_content_fts ON chunks USING GIN (to_tsvector('simple', content));

CREATE TABLE IF NOT EXISTS embeddings (
    id BIGSERIAL PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES uploaded_documents(id) ON DELETE CASCADE,
    chunk_id TEXT REFERENCES chunks(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    vector_backend TEXT NOT NULL DEFAULT 'chroma',
    vector_ref TEXT NOT NULL,
    dimension INTEGER,
    normalized BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS summaries (
    id UUID PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES uploaded_documents(id) ON DELETE CASCADE,
    algorithm_key TEXT NOT NULL,
    algorithm_group TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    target_length_ratio INTEGER,
    word_count INTEGER,
    generation_config JSONB NOT NULL DEFAULT '{}',
    citations JSONB NOT NULL DEFAULT '[]',
    consistency JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evaluations (
    id UUID PRIMARY KEY,
    summary_id UUID REFERENCES summaries(id) ON DELETE CASCADE,
    document_id TEXT NOT NULL REFERENCES uploaded_documents(id) ON DELETE CASCADE,
    rouge1 DOUBLE PRECISION,
    rouge2 DOUBLE PRECISION,
    rougel DOUBLE PRECISION,
    bleu DOUBLE PRECISION,
    bertscore_f1 DOUBLE PRECISION,
    semantic_similarity DOUBLE PRECISION,
    factual_consistency DOUBLE PRECISION,
    compression_ratio DOUBLE PRECISION,
    latency_seconds DOUBLE PRECISION,
    memory_usage_mb DOUBLE PRECISION,
    metrics JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS generated_reports (
    id UUID PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES uploaded_documents(id) ON DELETE CASCADE,
    report_type TEXT NOT NULL,
    title TEXT,
    content JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS quizzes (
    id UUID PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES uploaded_documents(id) ON DELETE CASCADE,
    difficulty TEXT,
    questions JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS podcasts (
    id UUID PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES uploaded_documents(id) ON DELETE CASCADE,
    script JSONB NOT NULL,
    audio_uri TEXT,
    tts_provider TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS research_runs (
    id UUID PRIMARY KEY,
    document_id TEXT REFERENCES uploaded_documents(id) ON DELETE SET NULL,
    run_type TEXT NOT NULL,
    config JSONB NOT NULL DEFAULT '{}',
    results JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
