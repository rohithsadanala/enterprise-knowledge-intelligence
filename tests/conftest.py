"""Shared fixtures: an isolated pipeline backed by fake embeddings and
the mock LLM, so the whole suite runs offline with no API key.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make the src layout importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eki.config import Settings  # noqa: E402
from eki.pipeline import KnowledgePipeline  # noqa: E402


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=str(tmp_path / "data"),
        embedding_provider="fake",
        llm_provider="mock",
        chunk_size=400,
        chunk_overlap=50,
        top_k=6,
        rerank_top_n=3,
        use_reranker=False,  # cross-encoder needs a model download; off in tests
    )


@pytest.fixture()
def pipeline(settings) -> KnowledgePipeline:
    pipe = KnowledgePipeline(settings)
    yield pipe
    pipe.close()


@pytest.fixture()
def sample_texts() -> dict[str, str]:
    return {
        "remote-work": (
            "Remote Work Policy. Employees may work remotely up to three days "
            "per week. Teams must agree on at least two common in-office days "
            "per week. Acme provides a laptop, monitor, and a one-time "
            "home-office stipend of $500. Remote employees must be available "
            "during core hours, 10:00 AM to 3:00 PM."
        ),
        "pto": (
            "Paid Time Off Policy. Full-time employees receive 20 days of paid "
            "time off per calendar year. PTO accrues monthly at a rate of 1.67 "
            "days per month. Up to 5 unused PTO days may be carried over into "
            "the next calendar year."
        ),
        "security": (
            "IT Security Policy. All employees must use a password manager and "
            "enable multi-factor authentication on every company system. "
            "Passwords must be at least 14 characters long. Suspected phishing "
            "emails must be reported within 1 hour of receipt."
        ),
    }


@pytest.fixture()
def seeded_pipeline(pipeline, sample_texts) -> KnowledgePipeline:
    for title, text in sample_texts.items():
        pipeline.ingest_text(title=title, text=text, source=f"{title}.txt")
    return pipeline


@pytest.fixture()
def api_client(seeded_pipeline):
    from fastapi.testclient import TestClient

    from eki.api import create_app

    app = create_app(pipeline=seeded_pipeline)
    return TestClient(app)
