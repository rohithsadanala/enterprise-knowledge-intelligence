"""Vector stores: FAISS (default, local) and ChromaDB (optional).

Both stores share one interface so the retrieval layer never knows
which backend is in use:

- ``add(chunks, vectors)`` — index a batch of chunks + embeddings
- ``search(query_vector, k)`` — return ``(chunk_ids, scores)`` ranked
  by cosine similarity (inner product on L2-normalised vectors)
- ``save(path)`` / ``load(path)`` — persist and restore the index
- ``__len__`` — number of indexed chunks

FAISS is the default: it is fast, has no server, and persists to a
single file. ChromaDB is available as an alternative when
``requirements-optional.txt`` is installed; pgvector is documented as a
deployment option in the README (same interface, Postgres-backed).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from eki.models import Chunk


class FaissStore:
    """In-memory FAISS index (IndexFlatIP) with file persistence.

    Vectors are L2-normalised on the way in, so inner product equals
    cosine similarity. Chunk ids are kept in a sidecar JSON file next
    to the ``.faiss`` index so ``search`` can map positions back to ids.
    """

    def __init__(self, dim: int) -> None:
        import faiss

        self.dim = dim
        self._faiss = faiss
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        self.index.add(arr)
        self.ids.extend(c.id for c in chunks)

    def search(self, query_vector: list[float], k: int) -> tuple[list[str], list[float]]:
        if len(self.ids) == 0:
            return [], []
        k = min(k, len(self.ids))
        q = np.asarray([query_vector], dtype=np.float32)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) or 1.0)
        scores, positions = self.index.search(q, k)
        ids = [self.ids[int(p)] for p in positions[0] if p != -1]
        return ids, [float(s) for s in scores[0][: len(ids)]]

    def remove_ids(self, ids_to_remove: set[str]) -> None:
        """Remove chunks by id (rebuilds the index — fine at this scale)."""
        keep = [i for i, cid in enumerate(self.ids) if cid not in ids_to_remove]
        if len(keep) == len(self.ids):
            return
        # Rebuild is handled by the pipeline (it re-adds surviving vectors);
        # here we only drop the ids so a fresh index can be rebuilt.
        self.ids = [self.ids[i] for i in keep]
        self.index = self._faiss.IndexFlatIP(self.dim)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(directory / "index.faiss"))
        (directory / "ids.json").write_text(json.dumps(self.ids))

    @classmethod
    def load(cls, directory: str | Path, dim: int) -> "FaissStore":
        import faiss

        directory = Path(directory)
        store = cls(dim)
        index_path = directory / "index.faiss"
        ids_path = directory / "ids.json"
        if index_path.exists() and ids_path.exists():
            store.index = faiss.read_index(str(index_path))
            store.ids = json.loads(ids_path.read_text())
        return store

    def __len__(self) -> int:
        return len(self.ids)


class ChromaStore:
    """ChromaDB-backed store (optional dependency).

    Same interface as :class:`FaissStore`. The collection is persisted
    to ``directory`` so it survives restarts.
    """

    def __init__(self, dim: int, directory: str | Path, collection: str = "eki") -> None:
        import chromadb

        self.dim = dim
        self.client = chromadb.PersistentClient(path=str(directory))
        self.collection = self.client.get_or_create_collection(name=collection)
        self._ids: list[str] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=[list(map(float, v)) for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=[{k: str(v) for k, v in c.metadata.items()} for c in chunks],
        )
        self._ids.extend(c.id for c in chunks)

    def search(self, query_vector: list[float], k: int) -> tuple[list[str], list[float]]:
        if self.collection.count() == 0:
            return [], []
        result = self.collection.query(
            query_embeddings=[list(map(float, query_vector))],
            n_results=min(k, self.collection.count()),
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        # Chroma returns L2 distance; convert to a similarity score.
        scores = [1.0 / (1.0 + float(d)) for d in distances]
        return list(ids), scores

    def remove_ids(self, ids_to_remove: set[str]) -> None:
        if ids_to_remove:
            self.collection.delete(ids=list(ids_to_remove))
        self._ids = [i for i in self._ids if i not in ids_to_remove]

    def save(self, directory: str | Path) -> None:
        # ChromaDB persists itself to its own directory on every write.
        return None

    @classmethod
    def load(cls, directory: str | Path, dim: int) -> "ChromaStore":
        return cls(dim, directory)

    def __len__(self) -> int:
        return int(self.collection.count())


def get_vector_store(kind: str, dim: int, directory: str | Path):
    """Factory: build the vector store named in the settings."""
    kind = kind.lower()
    if kind == "faiss":
        return FaissStore(dim)
    if kind == "chroma":
        return ChromaStore(dim, directory)
    raise ValueError(f"Unknown vector store: {kind!r} (expected 'faiss' or 'chroma')")
