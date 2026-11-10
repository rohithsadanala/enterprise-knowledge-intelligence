"""Tests for retrieval: BM25, RRF fusion, and the hybrid retriever."""

from eki.embeddings import FakeEmbeddings
from eki.models import Chunk
from eki.retrieval import BM25Index, HybridRetriever, expand_query, reciprocal_rank_fusion


def _chunks() -> list[Chunk]:
    return [
        Chunk(id="c1", doc_id="remote-work", text="Employees may work remotely up to three days per week."),
        Chunk(id="c2", doc_id="pto", text="Full-time employees receive 20 days of paid time off per year."),
        Chunk(id="c3", doc_id="security", text="Passwords must be at least 14 characters long and unique."),
        Chunk(id="c4", doc_id="security", text="Suspected phishing emails must be reported within 1 hour."),
    ]


def test_expand_query_drops_stopwords():
    terms = expand_query("What is the remote work policy?")
    assert "remote" in terms
    assert "work" in terms
    assert "what" not in terms
    assert "is" not in terms


def test_bm25_ranks_relevant_chunk_first():
    index = BM25Index()
    index.build(_chunks())
    results = index.search("how many PTO days per year", k=2)
    assert results
    assert results[0][0] == "c2"


def test_bm25_empty_index():
    index = BM25Index()
    assert index.search("anything", k=3) == []


def test_rrf_fusion_prefers_chunks_ranked_high_by_both():
    dense = [("c1", 0.9), ("c2", 0.8), ("c3", 0.7)]
    bm25 = [("c2", 12.0), ("c1", 9.0), ("c4", 3.0)]
    fused = reciprocal_rank_fusion([dense, bm25], weights=[0.6, 0.4])
    ids = [cid for cid, _ in fused]
    # c1 and c2 are top-2 in both lists, so they must lead the fusion.
    assert set(ids[:2]) == {"c1", "c2"}


def test_hybrid_retriever_returns_relevant_chunks(seeded_pipeline):
    chunks = seeded_pipeline.retriever.retrieve(
        "How many PTO days do employees get?", top_k=4, rerank_top_n=3
    )
    assert chunks
    assert len(chunks) <= 3
    # Doc ids are "<title>-<hash>", so match on the title prefix.
    assert any(c.doc_id.startswith("pto") for c in chunks)
    assert any("20 days" in c.text for c in chunks)


def test_hybrid_retriever_metadata_filter(seeded_pipeline):
    chunks = seeded_pipeline.retriever.retrieve(
        "policy",
        top_k=6,
        rerank_top_n=5,
        filters={"file_type": ".txt"},
    )
    # All seeded docs were ingested as .txt, so filtering keeps results.
    assert chunks
    chunks_none = seeded_pipeline.retriever.retrieve(
        "policy",
        top_k=6,
        rerank_top_n=5,
        filters={"file_type": ".pdf"},
    )
    assert chunks_none == []


def test_hybrid_retriever_empty_index(pipeline):
    assert pipeline.retriever.retrieve("anything", top_k=4, rerank_top_n=2) == []
