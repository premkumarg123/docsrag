"""
RAGPipeline: the central coordinator used by API routes and the eval harness.

Wires together: ingestion → embedding → storage → BM25 → hybrid search
→ reranking → generation.
"""

from __future__ import annotations

from collections.abc import Generator

from generation.generator import RAGGenerator
from ingestion.chunker import RecursiveChunker
from ingestion.embedder import Embedder
from ingestion.loader import DocumentLoader
from retrieval.bm25_index import BM25Index
from retrieval.hybrid_search import HybridSearcher
from retrieval.query_rewriter import QueryRewriter
from retrieval.reranker import CrossEncoderReranker
from retrieval.vector_store import VectorStore


class RAGPipeline:
    def __init__(
        self,
        dsn: str,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
    ) -> None:
        self._dsn = dsn
        self.loader = DocumentLoader()
        self.chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = Embedder()
        self.vector_store = VectorStore(dsn)
        self.bm25 = BM25Index()
        self.searcher = HybridSearcher(self.vector_store, self.bm25)
        self.reranker = CrossEncoderReranker()
        self.query_rewriter = QueryRewriter()
        self.generator = RAGGenerator()
        self._top_k_retrieve = top_k_retrieve
        self._top_k_rerank = top_k_rerank

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        self.vector_store.connect()
        self.rebuild_bm25()

    def shutdown(self) -> None:
        self.vector_store.close()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_text(
        self,
        text: str,
        name: str,
        metadata: dict | None = None,
    ) -> tuple[int, int]:
        doc = self.loader.load_text(text, name=name)
        return self._ingest_doc(doc, metadata)

    def ingest_file(
        self,
        path: str,
        name: str | None = None,
        metadata: dict | None = None,
    ) -> tuple[int, int]:
        doc = self.loader.load(path)
        if name:
            doc.name = name
        return self._ingest_doc(doc, metadata)

    def _ingest_doc(self, doc, metadata: dict | None) -> tuple[int, int]:
        chunks = self.chunker.chunk(doc.content, metadata=metadata or {})
        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed(texts)

        doc_id = self.vector_store.upsert_document(
            name=doc.name,
            source_uri=doc.source_uri,
            mime_type=doc.mime_type,
            metadata=metadata or {},
        )
        chunk_dicts = [
            {
                "chunk_index": c.chunk_index,
                "content": c.content,
                "embedding": emb,
                "token_count": c.token_count,
                "metadata": c.metadata,
            }
            for c, emb in zip(chunks, embeddings)
        ]
        self.vector_store.insert_chunks(doc_id, chunk_dicts)
        self.rebuild_bm25()
        return doc_id, len(chunks)

    def rebuild_bm25(self) -> None:
        """Rebuild in-memory BM25 index from all stored chunks."""
        with self.vector_store._cursor() as cur:
            cur.execute("SELECT id AS chunk_id, content FROM chunks")
            rows = cur.fetchall()
        self.bm25.build([dict(r) for r in rows])

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        rewrite: bool = True,
        rerank: bool = True,
    ) -> list[dict]:
        k = top_k or self._top_k_retrieve

        if rewrite:
            q = self.query_rewriter.rewrite(question)
        else:
            q = question

        q_emb = self.embedder.embed_one(q)
        results = self.searcher.search(q_emb, q, top_k=k)

        if rerank:
            reranked = self.reranker.rerank(q, results, top_k=self._top_k_rerank)
            return [
                {"chunk_id": r.chunk_id, "content": r.content, "score": r.rerank_score}
                for r in reranked
            ]

        return [
            {"chunk_id": r.chunk_id, "content": r.content, "score": r.rrf_score}
            for r in results[: self._top_k_rerank]
        ]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def answer(self, question: str, chunks: list[dict]) -> str:
        result = self.generator.generate(question, chunks)
        return result.answer

    def query(
        self,
        question: str,
        top_k: int = 5,
        rewrite: bool = True,
        rerank: bool = True,
    ) -> dict:
        rewritten = self.query_rewriter.rewrite(question) if rewrite else None
        chunks = self.retrieve(question, top_k=top_k, rewrite=rewrite, rerank=rerank)
        generated = self.generator.generate(question, chunks)
        return {
            "chunks": chunks,
            "answer": generated,
            "rewritten_question": rewritten,
        }

    def stream_answer(
        self,
        question: str,
        top_k: int = 5,
        rewrite: bool = True,
        rerank: bool = True,
    ) -> Generator[str, None, None]:
        chunks = self.retrieve(question, top_k=top_k, rewrite=rewrite, rerank=rerank)
        yield from self.generator.stream(question, chunks)
