"""FastAPI backend for Enterprise Knowledge Intelligence.

Endpoints:

- ``GET /health`` — liveness + index stats.
- ``GET /`` — the single-page demo UI (served from ``frontend/``).
- ``POST /ingest`` — upload one or more files (PDF / DOCX / MD / TXT).
- ``POST /ingest/text`` — index raw text as a document.
- ``POST /query`` — ask a question; returns the answer, citations and
  the query plan.
- ``GET /documents`` — list indexed documents.
- ``DELETE /documents/{doc_id}`` — remove a document and its chunks.

Run with::

    uvicorn eki.api:app --reload --port 8000
    # or
    python -m eki.cli serve
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from eki.pipeline import build_pipeline

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


# -- schemas ---------------------------------------------------------------------
class IngestTextRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1)
    source: str = "text"
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    filters: dict[str, Any] | None = None


class CitationOut(BaseModel):
    doc_id: str
    doc_title: str
    chunk_id: str
    score: float | None = None
    snippet: str = ""
    page: int | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationOut]
    query_plan: dict[str, Any]
    latency_ms: float
    retrieved_count: int


class DocumentOut(BaseModel):
    id: str
    title: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# -- app -------------------------------------------------------------------------
def create_app(pipeline=None) -> FastAPI:
    app = FastAPI(
        title="Enterprise Knowledge Intelligence",
        description="Advanced RAG: hybrid retrieval, reranking, LangGraph planning, cited answers.",
        version="0.1.0",
    )
    app.state.pipeline = pipeline or build_pipeline()

    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/health")
    def health() -> dict[str, Any]:
        stats = app.state.pipeline.stats()
        return {"status": "ok", "version": "0.1.0", **stats}

    @app.get("/")
    def index():
        index_path = FRONTEND_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"message": "Enterprise Knowledge Intelligence API — see /docs"}

    @app.post("/ingest")
    async def ingest(files: list[UploadFile] = File(...)) -> dict[str, Any]:
        pipeline = app.state.pipeline
        results: list[dict[str, Any]] = []
        uploads_dir = Path(pipeline.data_dir) / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        for upload in files:
            suffix = Path(upload.filename or "upload").suffix.lower()
            if suffix not in {".pdf", ".docx", ".md", ".markdown", ".txt"}:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {suffix!r} (file: {upload.filename})",
                )
            dest = uploads_dir / (upload.filename or "upload")
            content = await upload.read()
            dest.write_bytes(content)
            try:
                results.append(pipeline.ingest_file(dest))
            except Exception as exc:
                raise HTTPException(
                    status_code=422, detail=f"Failed to ingest {upload.filename}: {exc}"
                ) from exc
        return {"ingested": len(results), "results": results}

    @app.post("/ingest/text")
    def ingest_text(req: IngestTextRequest) -> dict[str, Any]:
        result = app.state.pipeline.ingest_text(
            title=req.title, text=req.text, source=req.source, metadata=req.metadata
        )
        return {"ingested": 1, "results": [result]}

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        result = app.state.pipeline.query(
            req.question, top_k=req.top_k, filters=req.filters
        )
        return QueryResponse(
            question=result.question,
            answer=result.answer,
            citations=[CitationOut(**vars(c)) for c in result.citations],
            query_plan=result.query_plan,
            latency_ms=round(result.latency_ms, 1),
            retrieved_count=result.retrieved_count,
        )

    @app.get("/documents", response_model=list[DocumentOut])
    def list_documents() -> list[DocumentOut]:
        return [
            DocumentOut(id=d.id, title=d.title, source=d.source, metadata=d.metadata)
            for d in app.state.pipeline.list_documents()
        ]

    @app.delete("/documents/{doc_id}")
    def delete_document(doc_id: str) -> dict[str, Any]:
        if not app.state.pipeline.delete_document(doc_id):
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
        return {"deleted": doc_id}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "eki.api:app",
        host=os.environ.get("EKI_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("EKI_API_PORT", "8000")),
        reload=False,
    )
