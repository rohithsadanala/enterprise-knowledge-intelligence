"""Embedding providers.

All providers share one interface — ``embed(texts) -> list[list[float]]``
— so the pipeline never cares which model is behind it.

``fake``
    Deterministic, offline, no downloads: each text is hashed into a
    fixed-size vector (seeded per token). Good enough for tests, demos
    and CI, and it preserves the property that identical texts embed
    identically and similar texts (shared tokens) embed closer.

``huggingface``
    Sentence-Transformers (default ``all-MiniLM-L6-v2``, 384 dims).
    Needs ``requirements-optional.txt``.

``openai``
    ``text-embedding-3-small`` via LangChain. Needs ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class FakeEmbeddings:
    """Deterministic hash-based embeddings (offline, no model download).

    Each token contributes to a fixed number of pseudo-random buckets,
    so texts sharing vocabulary end up closer in cosine space than
    unrelated texts — enough for retrieval tests and the demo.
    """

    def __init__(self, dim: int = 384, seed: int = 42) -> None:
        self.dim = dim
        self.seed = seed

    def _token_vector(self, token: str) -> list[float]:
        digest = hashlib.sha256(f"{self.seed}:{token}".encode()).digest()
        # 8 bytes -> 4 float buckets per hash, chained for the full dim.
        vec = [0.0] * self.dim
        for i in range(self.dim):
            chunk = digest[(i * 2) % len(digest)] / 255.0
            # Alternate sign deterministically so the mean stays ~0.
            vec[i] = chunk if i % 2 == 0 else -chunk
        return vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            tokens = _TOKEN.findall(text.lower())
            vec = [0.0] * self.dim
            for token in set(tokens):
                tv = self._token_vector(token)
                for i in range(self.dim):
                    vec[i] += tv[i]
            out.append(_l2_normalize(vec))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class HuggingFaceEmbeddings:
    """Sentence-Transformers embeddings (local, no API key)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in np.asarray(vectors)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class OpenAIEmbeddings:
    """OpenAI embeddings via LangChain (needs ``OPENAI_API_KEY``)."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = "") -> None:
        from langchain_openai import OpenAIEmbeddings as _OE

        self.model = model
        self.client = _OE(model=model, api_key=api_key or None)
        self.dim = 1536 if "small" in model else 3072

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self.client.embed_documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(map(float, self.client.embed_query(text)))


def get_embeddings(provider: str, model: str = "", dim: int = 384, api_key: str = ""):
    """Factory: build the embedding provider named in the settings."""
    provider = provider.lower()
    if provider == "huggingface":
        return HuggingFaceEmbeddings(model or "sentence-transformers/all-MiniLM-L6-v2")
    if provider == "openai":
        return OpenAIEmbeddings(model or "text-embedding-3-small", api_key=api_key)
    if provider == "fake":
        return FakeEmbeddings(dim=dim)
    raise ValueError(
        f"Unknown embedding provider: {provider!r} "
        "(expected 'fake', 'huggingface' or 'openai')"
    )
