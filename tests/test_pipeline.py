"""End-to-end tests: ingest a folder, ask questions, check citations."""

from pathlib import Path

from eki.evaluation import evaluate_case, load_eval_set, summarize
from eki.models import QueryResult


def test_ingest_directory_and_query(seeded_pipeline, tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "travel.txt").write_text(
        "Travel Policy. All travel bookings need manager approval. "
        "Flights must be booked at least 14 days in advance."
    )
    results = seeded_pipeline.ingest_directory(docs_dir)
    assert len(results) == 1
    assert results[0]["chunks"] >= 1

    result = seeded_pipeline.query("How far in advance must flights be booked?")
    assert "14 days" in result.answer
    assert result.citations
    assert result.retrieved_count >= 1


def test_query_answer_is_grounded_in_retrieved_text(seeded_pipeline):
    result = seeded_pipeline.query("What is the minimum password length?")
    assert "14" in result.answer
    # The answer must cite the security document.
    assert any(c.doc_id.startswith("security") for c in result.citations)


def test_query_plan_classifies_comparison(seeded_pipeline):
    result = seeded_pipeline.query("Compare the PTO policy and the remote work policy.")
    assert result.query_plan["query_type"] == "comparison"
    assert len(result.query_plan["sub_questions"]) >= 1


def test_delete_document_removes_it_from_results(seeded_pipeline):
    docs = seeded_pipeline.list_documents()
    pto = next(d for d in docs if d.title == "pto")
    assert seeded_pipeline.delete_document(pto.id) is True
    result = seeded_pipeline.query("How many PTO days do employees get?")
    assert not any(c.doc_id.startswith("pto") for c in result.citations)


def test_eval_harness_runs_on_sample_set(settings):
    from eki.pipeline import KnowledgePipeline

    # Use the real sample docs (not the conftest texts) so the eval set's
    # relevant-doc stems match the indexed titles.
    repo = Path(__file__).resolve().parent.parent
    pipe = KnowledgePipeline(settings)
    pipe.ingest_directory(repo / "docs" / "sample")
    try:
        cases = load_eval_set(repo / "eval_data" / "eval_set.json")
        assert len(cases) == 8
        title_to_id = {d.title: d.id for d in pipe.list_documents()}
        results = []
        for case in cases:
            case.relevant_doc_ids = [title_to_id.get(s, s) for s in case.relevant_doc_ids]
            result = pipe.query(case.question)
            chunk_texts = [
                pipe.chunks[c.chunk_id].text
                for c in result.citations
                if c.chunk_id in pipe.chunks
            ]
            results.append(evaluate_case(case, result, chunk_texts, k=3))
        summary = summarize(results)
        assert summary["n"] == 8
        # The sample set is written against the sample docs, so retrieval
        # should be strong and answers grounded.
        assert summary["mrr"] >= 0.5
        assert summary["faithfulness"] >= 0.5
    finally:
        pipe.close()
