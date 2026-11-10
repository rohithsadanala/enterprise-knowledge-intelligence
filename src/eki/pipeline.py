"""The RAG pipeline: ingestion + question answering.

:class:`KnowledgePipeline` is the single entry point used by the API,
the CLI and the eval harness. It owns:

- the embedding provider and the LLM (both pluggable),
- the SQLite docstore (source of truth),
- the FAISS/Chroma vector store and the hybrid retriever,
- the LangGraph query planner.

Flow for ``ingest``:
    load files -> chunk -> embed -> store chunks + embeddings ->
    add to vector store -> rebuild BM25 -> persist.

Flow for ``query``:
    plan (LangGraph) -> hybrid retrieve (dense + BM25, RRF, rerank)
    -> generate (LangChain chain, cited) -> attach citations.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from eki import config as config_mod
from eki.chunking import chunk_document
from eki.config import Settings
from eki.documents import ensure_data_dir, load_directory, load_document
from eki.embeddings import get_embeddings
from eki.generation import all_citations, build_citations, generate_answer
from eki.llm import get_llm
from eki.models import Chunk, Document, QueryResult
from eki.planning import plan_query
from eki.retrieval import CrossEncoderReranker, HybridRetriever
from eki.store import DocStore
from eki.vector_store import get_vector_store


class KnowledgePipeline:
    """End-to-end RAG pipeline (ingest + query)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.data_dir = ensure_data_dir(self.settings.data_dir)

        self.embedder = get_embeddings(
            self.settings.embedding_provider,
            model=self.settings.embedding_model,
            dim=self.settings.embedding_dim,
            api_key=self.settings.openai_api_key,
        )
        self.llm = get_llm(
            self.settings.llm_provider,
            model=self.settings.llm_model,
            api_key=self.settings.openai_api_key,
            temperature=self.settings.llm_temperature,
        )
        self.store = DocStore(self.data_dir / "docstore.db")
        self.vector_store = get_vector_store(
            self.settings.vector_store,
            dim=self._embedding_dim(),
            directory=self.data_dir / "vector_store",
        )
        self.reranker = CrossEncoderReranker(
            self.settings.reranker_model, enabled=self.settings.use_reranker
        )
        self._load_index()

    # -- setup -------------------------------------------------------------------
    def _embedding_dim(self) -> int:
        return int(getattr(self.embedder, "dim", self.settings.embedding_dim))

    def _load_index(self) -> None:
        """Rebuild the vector store + retriever from the persisted docstore."""
        pairs = self.store.list_chunks_with_embeddings()
        self.chunks = {c.id: c for c, _ in pairs}
        self._rebuild_vector_store(pairs)
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            chunks=self.chunks,
            dense_weight=self.settings.dense_weight,
            bm25_weight=self.settings.bm25_weight,
            reranker=self.reranker,
        )

    def _rebuild_vector_store(
        self, pairs: list[tuple[Chunk, list[float]]] | None = None
    ) -> None:
        """Recreate the vector store from the docstore (source of truth).

        Used at startup and after deletions, so the index can never
        drift out of sync with the stored chunks.
        """
        if pairs is None:
            pairs = self.store.list_chunks_with_embeddings()
        self.vector_store = get_vector_store(
            self.settings.vector_store,
            dim=self._embedding_dim(),
            directory=self.data_dir / "vector_store",
        )
        if pairs:
            chunks, vectors = zip(*pairs)
            self.vector_store.add(list(chunks), [list(v) for v in vectors])
            self.chunks = {c.id: c for c in chunks}

    def _refresh_retriever(self) -> None:
        self.retriever.chunks = self.chunks
        self.retriever.refresh()

    # -- ingestion ---------------------------------------------------------------
    def ingest_document(self, doc: Document) -> dict[str, Any]:
        """Chunk, embed and index one already-loaded document."""
        embed_fn = None
        if self.settings.chunking_strategy == "semantic":
            embed_fn = self.embedder.embed
        chunks = chunk_document(
            doc,
            strategy=self.settings.chunking_strategy,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            embed_fn=embed_fn,
        )
        if not chunks:
            return {"doc_id": doc.id, "title": doc.title, "chunks": 0}

        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.upsert_document(doc)
        self.store.upsert_chunks(doc.id, chunks, vectors)
        for chunk in chunks:
            self.chunks[chunk.id] = chunk
        self.vector_store.add(chunks, vectors)
        self._refresh_retriever()
        self._persist()
        return {"doc_id": doc.id, "title": doc.title, "chunks": len(chunks)}

    def ingest_file(self, path: str | Path) -> dict[str, Any]:
        """Load and index a single file (PDF, DOCX, MD or TXT)."""
        doc = load_document(path)
        return self.ingest_document(doc)

    def ingest_directory(self, directory: str | Path) -> list[dict[str, Any]]:
        """Load and index every supported file under a directory."""
        results = []
        for doc in load_directory(directory):
            results.append(self.ingest_document(doc))
        return results

    def ingest_text(
        self, title: str, text: str, source: str = "text", metadata: dict | None = None
    ) -> dict[str, Any]:
        """Index raw text as a document (used by the API's /ingest/text)."""
        import hashlib

        digest = hashlib.sha256(text.encode()).hexdigest()[:12]
        doc = Document(
            id=f"{title}-{digest}",
            title=title,
            source=source,
            content=text,
            metadata=metadata or {"file_type": ".txt"},
        )
        return self.ingest_document(doc)

    # -- documents ---------------------------------------------------------------
    def list_documents(self) -> list[Document]:
        return self.store.list_documents()

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document and its chunks from every index."""
        if self.store.get_document(doc_id) is None:
            return False
        self.store.delete_document(doc_id)
        self.chunks = {
            cid: c for cid, c in self.chunks.items() if c.doc_id != doc_id
        }
        # Rebuild the vector store from the surviving chunks so the
        # index can never drift out of sync with the docstore.
        self._rebuild_vector_store()
        self.retriever.vector_store = self.vector_store
        self._refresh_retriever()
        self._persist()
        return True

    # -- querying ------------------------------------------------------------------
    def query(
        self,
        question: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Answer a question against the indexed knowledge base."""
        started = time.perf_counter()
        if not question.strip():
            return QueryResult(
                question=question,
                answer="Please ask a question.",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        plan = plan_query(question, llm=self.llm, base_top_k=top_k or self.settings.top_k)
        k = int(plan.get("top_k", top_k or self.settings.top_k))

        chunks = self.retriever.retrieve(
            plan["rewritten_query"],
            top_k=k,
            rerank_top_n=self.settings.rerank_top_n,
            filters=filters,
            expanded_terms=plan.get("expansion_terms") or [],
        )

        answer = generate_answer(self.llm, question, chunks)
        citations = build_citations(chunks, answer)
        if not citations:
            # The model did not cite: attach the retrieved sources anyway
            # so the answer is never presented without provenance.
            citations = all_citations(chunks)

        return QueryResult(
            question=question,
            answer=answer,
            citations=citations,
            query_plan=plan,
            latency_ms=(time.perf_counter() - started) * 1000,
            retrieved_count=len(chunks),
        )

    # -- persistence ---------------------------------------------------------------
    def _persist(self) -> None:
        if hasattr(self.vector_store, "save"):
            try:
                self.vector_store.save(self.data_dir / "vector_store")
            except Exception:
                pass  # ChromaDB persists itself; FAISS save failures are non-fatal here

    def stats(self) -> dict[str, Any]:
        return {
            "documents": len(self.store.list_documents()),
            "chunks": self.store.count_chunks(),
            "indexed_vectors": len(self.vector_store),
            "embedding_provider": self.settings.embedding_provider,
            "llm_provider": self.settings.llm_provider,
            "vector_store": self.settings.vector_store,
            "reranker_available": self.reranker.available if self.reranker.enabled else False,
        }

    def close(self) -> None:
        self.store.close()


def build_pipeline(settings: Settings | None = None) -> KnowledgePipeline:
    """Convenience constructor (also loads ``.env`` if present)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    return KnowledgePipeline(settings or Settings.from_env())
