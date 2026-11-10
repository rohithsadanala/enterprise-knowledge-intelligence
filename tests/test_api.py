"""Tests for the FastAPI endpoints (health, ingest, query, documents)."""

from fastapi.testclient import TestClient


def test_health(api_client: TestClient):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["documents"] == 3
    assert body["chunks"] >= 3
    assert body["embedding_provider"] == "fake"
    assert body["llm_provider"] == "mock"


def test_list_documents(api_client: TestClient):
    resp = api_client.get("/documents")
    assert resp.status_code == 200
    titles = {d["title"] for d in resp.json()}
    assert titles == {"remote-work", "pto", "security"}


def test_ingest_text(api_client: TestClient):
    resp = api_client.post(
        "/ingest/text",
        json={"title": "expenses", "text": "Expense reports must be submitted within 30 days of the purchase."},
    )
    assert resp.status_code == 200
    assert resp.json()["ingested"] == 1
    assert resp.json()["results"][0]["chunks"] >= 1

    docs = api_client.get("/documents").json()
    assert "expenses" in {d["title"] for d in docs}


def test_ingest_text_validation(api_client: TestClient):
    resp = api_client.post("/ingest/text", json={"title": "", "text": "x"})
    assert resp.status_code == 422


def test_query_returns_answer_with_citations(api_client: TestClient):
    resp = api_client.post("/query", json={"question": "How many PTO days do employees get?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["question"]
    assert body["answer"]
    assert "20 days" in body["answer"]
    # Every answer must carry citations.
    assert body["citations"], "answer has no citations"
    assert body["retrieved_count"] >= 1
    assert body["query_plan"]["query_type"] in {"factual", "summary", "comparison", "multi_hop"}
    for citation in body["citations"]:
        assert citation["doc_id"]
        assert citation["snippet"]


def test_query_empty_index_returns_graceful_answer():
    from eki.api import create_app
    from eki.config import Settings
    from eki.pipeline import KnowledgePipeline
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        pipe = KnowledgePipeline(Settings(data_dir=tmp, embedding_provider="fake",
                                         llm_provider="mock", use_reranker=False))
        client = TestClient(create_app(pipeline=pipe))
        resp = client.post("/query", json={"question": "Anything?"})
        assert resp.status_code == 200
        assert "don't have enough information" in resp.json()["answer"]
        pipe.close()


def test_query_validation(api_client: TestClient):
    resp = api_client.post("/query", json={"question": ""})
    assert resp.status_code == 422


def test_delete_document(api_client: TestClient):
    docs = api_client.get("/documents").json()
    doc_id = docs[0]["id"]
    resp = api_client.delete(f"/documents/{doc_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == doc_id
    remaining = {d["id"] for d in api_client.get("/documents").json()}
    assert doc_id not in remaining

    resp = api_client.delete(f"/documents/{doc_id}")
    assert resp.status_code == 404


def test_ingest_file_upload(api_client: TestClient):
    resp = api_client.post(
        "/ingest",
        files=[("files", ("note.txt", b"Travel bookings need manager approval.", "text/plain"))],
    )
    assert resp.status_code == 200
    assert resp.json()["ingested"] == 1
