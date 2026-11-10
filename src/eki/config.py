"""Configuration for Enterprise Knowledge Intelligence.

All settings come from environment variables (or a ``.env`` file loaded
by :mod:`eki.cli` / :mod:`eki.api`), so no secret ever lives in code.
See ``.env.example`` for the full list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _getenv(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _getint(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _getfloat(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _getbool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Runtime configuration. Defaults are chosen so the system runs
    fully offline (fake embeddings + mock LLM) with no API key."""

    # -- data -----------------------------------------------------------------
    data_dir: str = field(default_factory=lambda: _getenv("EKI_DATA_DIR", "./data"))
    """Directory holding the SQLite docstore, the FAISS index and uploads."""

    # -- embeddings -----------------------------------------------------------
    embedding_provider: str = field(
        default_factory=lambda: _getenv("EKI_EMBEDDING_PROVIDER", "fake")
    )
    """One of ``fake`` (deterministic, offline), ``huggingface`` or ``openai``."""
    embedding_model: str = field(
        default_factory=lambda: _getenv(
            "EKI_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    embedding_dim: int = field(default_factory=lambda: _getint("EKI_EMBEDDING_DIM", 384))
    """Vector size. 384 matches all-MiniLM-L6-v2 and the fake provider."""

    # -- LLM ------------------------------------------------------------------
    llm_provider: str = field(default_factory=lambda: _getenv("EKI_LLM_PROVIDER", "mock"))
    """One of ``mock`` (extractive, offline), ``openai`` or ``huggingface``."""
    llm_model: str = field(
        default_factory=lambda: _getenv("EKI_LLM_MODEL", "gpt-4o-mini")
    )
    openai_api_key: str = field(default_factory=lambda: _getenv("OPENAI_API_KEY", ""))
    llm_temperature: float = field(
        default_factory=lambda: _getfloat("EKI_LLM_TEMPERATURE", 0.0)
    )

    # -- chunking -------------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: _getint("EKI_CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _getint("EKI_CHUNK_OVERLAP", 120))
    chunking_strategy: str = field(
        default_factory=lambda: _getenv("EKI_CHUNKING_STRATEGY", "fixed")
    )
    """``fixed`` (size/overlap) or ``semantic`` (sentence-similarity splits)."""

    # -- retrieval ------------------------------------------------------------
    top_k: int = field(default_factory=lambda: _getint("EKI_TOP_K", 8))
    """Candidates fetched from each retriever before fusion."""
    rerank_top_n: int = field(default_factory=lambda: _getint("EKI_RERANK_TOP_N", 5))
    """Chunks kept after reranking and passed to the generator."""
    dense_weight: float = field(default_factory=lambda: _getfloat("EKI_DENSE_WEIGHT", 0.6))
    bm25_weight: float = field(default_factory=lambda: _getfloat("EKI_BM25_WEIGHT", 0.4))
    use_reranker: bool = field(default_factory=lambda: _getbool("EKI_USE_RERANKER", True))
    reranker_model: str = field(
        default_factory=lambda: _getenv(
            "EKI_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )
    vector_store: str = field(default_factory=lambda: _getenv("EKI_VECTOR_STORE", "faiss"))
    """``faiss`` (default, local) or ``chroma`` (needs requirements-optional)."""

    # -- api ------------------------------------------------------------------
    api_host: str = field(default_factory=lambda: _getenv("EKI_API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: _getint("EKI_API_PORT", 8000))

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from the current environment."""
        return cls()
