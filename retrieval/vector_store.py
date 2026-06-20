"""pgvector-backed store for chunk embeddings and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import psycopg2
import psycopg2.extras


@dataclass
class ChunkResult:
    chunk_id: int
    document_id: int
    content: str
    score: float
    metadata: dict


class VectorStore:
    """
    CRUD and ANN search over the `chunks` table using pgvector.

    Uses cosine similarity; embeddings must be L2-normalised before insert
    (cosine_ops + normalised vectors = dot-product, which is fastest on pgvector).
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn: Optional[psycopg2.extensions.connection] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._conn = psycopg2.connect(self._dsn)
        psycopg2.extras.register_uuid(self._conn)

    def close(self) -> None:
        if self._conn:
            self._conn.close()

    def _cursor(self):
        if not self._conn or self._conn.closed:
            self.connect()
        return self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert_document(
        self,
        name: str,
        source_uri: str | None = None,
        mime_type: str = "text/plain",
        metadata: dict | None = None,
    ) -> int:
        sql = """
            INSERT INTO documents (name, source_uri, mime_type, metadata)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """
        with self._cursor() as cur:
            cur.execute(sql, (name, source_uri, mime_type, psycopg2.extras.Json(metadata or {})))
            doc_id = cur.fetchone()["id"]
        self._conn.commit()
        return doc_id

    def insert_chunks(
        self,
        document_id: int,
        chunks: List[dict],  # {content, embedding, chunk_index, token_count, metadata}
    ) -> List[int]:
        sql = """
            INSERT INTO chunks (document_id, chunk_index, content, embedding, token_count, metadata)
            VALUES %s
            RETURNING id
        """
        records = [
            (
                document_id,
                c["chunk_index"],
                c["content"],
                c["embedding"],
                c.get("token_count", 0),
                psycopg2.extras.Json(c.get("metadata", {})),
            )
            for c in chunks
        ]
        with self._cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, records, fetch=True)
            ids = [row["id"] for row in cur.fetchall()]
        self._conn.commit()
        return ids

    def delete_document(self, document_id: int) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        document_id: int | None = None,
    ) -> List[ChunkResult]:
        """Return top-k chunks ordered by cosine similarity (highest first)."""
        filter_clause = "AND c.document_id = %s" if document_id else ""
        params: list = [str(query_embedding), top_k]
        if document_id:
            params.insert(1, document_id)

        sql = f"""
            SELECT
                c.id              AS chunk_id,
                c.document_id,
                c.content,
                1 - (c.embedding <=> %s::vector) AS score,
                c.metadata
            FROM chunks c
            {filter_clause}
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """
        params = [str(query_embedding), str(query_embedding), top_k]
        if document_id:
            sql = sql.replace(filter_clause, "AND c.document_id = %s")
            params = [str(query_embedding), document_id, str(query_embedding), top_k]

        with self._cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            ChunkResult(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                content=r["content"],
                score=float(r["score"]),
                metadata=r["metadata"] or {},
            )
            for r in rows
        ]

    def get_chunk(self, chunk_id: int) -> ChunkResult | None:
        with self._cursor() as cur:
            cur.execute(
                "SELECT id, document_id, content, metadata FROM chunks WHERE id = %s",
                (chunk_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return ChunkResult(
            chunk_id=row["id"],
            document_id=row["document_id"],
            content=row["content"],
            score=1.0,
            metadata=row["metadata"] or {},
        )

    def list_documents(self) -> List[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT id, name, source_uri, mime_type, metadata, created_at FROM documents ORDER BY created_at DESC"
            )
            return [dict(r) for r in cur.fetchall()]
