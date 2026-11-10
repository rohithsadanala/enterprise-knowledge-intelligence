"""Hybrid retrieval: BM25 + dense vectors, fused, reranked.

Pipeline for a query:

1. **Query rewriting / expansion** — the planner (see
   :mod:`eki.planning`) may rewrite the question into a search-friendly
   form and add expansion terms. A cheap lexical expansion (lowercase,
   stopword-stripped keywords) always runs.
2. **Dense retrieval** — top-k chunks from the FAISS/Chroma store.
3. **BM25 retrieval** — top-k chunks from the keyword index.
4. **Reciprocal Rank Fusion (RRF)** — merges the two ranked lists into
   one, weighting dense vs. sparse by the configured weights.
5. **Cross-encoder reranking** — a small cross-encoder scores each
   (query, chunk) pair directly and reorders the fused list. Falls back
   to the fused order when the model is unavailable (offline mode).
6. **Metadata filtering** — optional ``filters`` (e.g.
   ``{"doc_id": "..."}`` or ``{"file_type": ".pdf"}``) restrict the
   candidate pool before fusion.
"""

from __future__ import annotations

import re
from typing import Any

from eki.models import Chunk

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "which", "who",
    "whom", "how", "when", "where", "why", "does", "do", "did", "can",
    "could", "should", "would", "will", "shall", "may", "might", "of",
    "in", "on", "at", "to", "for", "with", "by", "from", "and", "or",
    "but", "not", "no", "yes", "if", "then", "than", "as", "it", "its",
    "this", "that", "these", "those", "there", "here", "about", "into",
    "over", "under", "between", "per", "via", "please", "tell", "me",
    "give", "show", "list", "explain", "describe",
}

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return _TOKEN.findall(text.lower())


def expand_query(question: str) -> list[str]:
    """Lexical expansion terms: content keywords from the question."""
    return [t for t in tokenize(question) if t not in _STOPWORDS]


class BM25Index:
    """Keyword index over chunks, backed by ``rank-bm25``."""

    def __init__(self) -> None:
        from rank_bm25 import BM25Okapi

        self._bm25_cls = BM25Okapi
        self._index = None
        self._ids: list[str] = []

    def build(self, chunks: list[Chunk]) -> None:
        corpus = [tokenize(c.text) for c in chunks]
        self._ids = [c.id for c in chunks]
        self._index = self._bm25_cls(corpus) if corpus else None

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._index is None or not self._ids:
            return []
        scores = self._index.get_scores(tokenize(query))
        ranked = sorted(
            zip(self._ids, scores), key=lambda pair: pair[1], reverse=True
        )
        return [(cid, float(s)) for cid, s in ranked[:k] if s > 0]

    def __len__(self) -> int:
        return len(self._ids)


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge ranked lists by RRF: ``score = sum(w / (k + rank))``.

    Each input list is ``[(chunk_id, score), ...]`` in rank order; the
    raw scores are ignored — only the rank matters, which is what makes
    RRF robust across retrievers with incomparable score scales.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    fused: dict[str, float] = {}
    for weight, ranked in zip(weights, ranked_lists):
        for rank, (cid, _score) in enumerate(ranked, start=1):
            fused[cid] = fused.get(cid, 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)


class CrossEncoderReranker:
    """Cross-encoder reranker (sentence-transformers), lazy-loaded.

    Scores (query, chunk) pairs directly — far more accurate than
    bi-encoder cosine for the final ordering. If the model cannot be
    loaded (offline / not installed), the reranker becomes a no-op
    that preserves the input order, and the pipeline logs a warning.
    """

    def __init__(self, model_name: str, enabled: bool = True) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self._model = None
        self._load_error: str | None = None

    def _load(self):
        if self._model is not None or self._load_error is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        except Exception as exc:  # noqa: BLE001 - offline fallback is by design
            self._load_error = str(exc)
        return self._model

    @property
    def available(self) -> bool:
        return self.enabled and self._load() is not None

    def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        """Reorder chunks by cross-encoder relevance; keep ``top_n``."""
        if not chunks:
            return []
        model = self._load() if self.enabled else None
        if model is None:
            return chunks[:top_n]
        pairs = [(query, c.text) for c in chunks]
        scores = model.predict(pairs)
        scored = sorted(zip(chunks, scores), key=lambda p: p[1], reverse=True)
        out: list[Chunk] = []
        for chunk, score in scored[:top_n]:
            chunk.score = float(score)
            out.append(chunk)
        return out


class HybridRetriever:
    """Dense + BM25 hybrid retriever with RRF fusion and reranking."""

    def __init__(
        self,
        vector_store,
        embedder,
        chunks: dict[str, Chunk],
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunks = chunks  # chunk_id -> Chunk
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.reranker = reranker
        self.bm25 = BM25Index()
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        self.bm25.build(list(self.chunks.values()))

    def refresh(self) -> None:
        """Rebuild the BM25 index after the chunk set changes."""
        self._rebuild_bm25()

    def _apply_filters(
        self, candidates: list[str], filters: dict[str, Any] | None
    ) -> list[str]:
        if not filters:
            return candidates
        out = []
        for cid in candidates:
            chunk = self.chunks.get(cid)
            if chunk is None:
                continue
            meta = {**chunk.metadata, "doc_id": chunk.doc_id}
            if all(str(meta.get(k, "")) == str(v) for k, v in filters.items()):
                out.append(cid)
        return out

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        rerank_top_n: int = 5,
        filters: dict[str, Any] | None = None,
        expanded_terms: list[str] | None = None,
    ) -> list[Chunk]:
        """Run the full hybrid retrieval for one query."""
        search_text = query
        if expanded_terms:
            search_text = query + " " + " ".join(expanded_terms)

        # 1. Dense retrieval.
        dense_ids, _dense_scores = self.vector_store.search(
            self.embedder.embed_query(query), k=top_k
        )
        dense_ranked = [(cid, 0.0) for cid in dense_ids]

        # 2. BM25 retrieval (uses the expanded query text).
        bm25_ranked = self.bm25.search(search_text, k=top_k)

        # 3. Metadata filtering before fusion.
        allowed = None
        if filters:
            allowed = set(self._apply_filters(
                [cid for cid, _ in dense_ranked + bm25_ranked], filters
            ))
            dense_ranked = [p for p in dense_ranked if p[0] in allowed]
            bm25_ranked = [p for p in bm25_ranked if p[0] in allowed]

        # 4. RRF fusion.
        fused = reciprocal_rank_fusion(
            [dense_ranked, bm25_ranked],
            weights=[self.dense_weight, self.bm25_weight],
        )

        # 5. Rerank (or keep fused order if the reranker is unavailable).
        candidates = [self.chunks[cid] for cid, _ in fused if cid in self.chunks]
        for chunk, (_cid, score) in zip(candidates, fused):
            chunk.score = float(score)
        if self.reranker is not None:
            return self.reranker.rerank(query, candidates, top_n=rerank_top_n)
        return candidates[:rerank_top_n]
