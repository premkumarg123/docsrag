-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table: one row per ingested source file
CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    source_uri  TEXT,
    mime_type   TEXT,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Chunks table: one row per retrievable chunk
CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    -- 384-dim for all-MiniLM-L6-v2
    embedding   vector(384),
    token_count INTEGER,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for approximate nearest-neighbour search.
-- Unlike IVFFlat, HNSW updates incrementally so it works correctly
-- even when the index is created before data is inserted.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Full-text search index for BM25 hybrid fallback
CREATE INDEX IF NOT EXISTS chunks_content_fts
    ON chunks USING gin(to_tsvector('english', content));

-- Eval runs table
CREATE TABLE IF NOT EXISTS eval_runs (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    config      JSONB DEFAULT '{}',
    results     JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
