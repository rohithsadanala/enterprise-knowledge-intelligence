"""Core data models for Enterprise Knowledge Intelligence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A single ingested source document."""

    id: str
    title: str
    source: str  # original file path or URI
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable slice of a document."""

    id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None  # retrieval score, set at query time


@dataclass
class Citation:
    """A source reference attached to a generated answer."""

    doc_id: str
    doc_title: str
    chunk_id: str
    score: float | None = None
    snippet: str = ""
    page: int | None = None

    def to_reference(self, index: int) -> str:
        """Render as ``[1] Title (chunk ...)`` for display."""
        label = f"[{index}] {self.doc_title}"
        if self.page is not None:
            label += f", p. {self.page}"
        return label


@dataclass
class QueryResult:
    """The full result of asking the knowledge base a question."""

    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    query_plan: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    retrieved_count: int = 0
