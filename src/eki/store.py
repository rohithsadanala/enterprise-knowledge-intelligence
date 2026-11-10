"""Persistent document store (SQLite, standard library only).

Two tables:

- ``documents`` — one row per ingested document (id, title, source,
  metadata JSON).
- ``chunks`` — one row per chunk (id, doc_id, text, metadata JSON,
  embedding JSON). The embedding is stored alongside the chunk so the
  FAISS index can be rebuilt after a restart without re-embedding.

The store is the source of truth for what is indexed; the FAISS index
and the BM25 index are derived from it and rebuilt on load.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from eki.models import Chunk, Document


class DocStore:
    """SQLite-backed store for documents, chunks and their embeddings."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI runs sync endpoints in a
        # worker thread pool, so the connection must be usable across
        # threads. The pipeline is single-writer in practice.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                embedding TEXT NOT NULL DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            """
        )
        self._conn.commit()

    # -- documents -------------------------------------------------------------
    def upsert_document(self, doc: Document) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO documents (id, title, source, metadata) "
            "VALUES (?, ?, ?, ?)",
            (doc.id, doc.title, doc.source, json.dumps(doc.metadata)),
        )
        self._conn.commit()

    def get_document(self, doc_id: str) -> Document | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return None
        return Document(
            id=row["id"],
            title=row["title"],
            source=row["source"],
            content="",  # content is reconstructed from chunks if needed
            metadata=json.loads(row["metadata"]),
        )

    def list_documents(self) -> list[Document]:
        rows = self._conn.execute("SELECT * FROM documents ORDER BY title").fetchall()
        return [
            Document(
                id=r["id"],
                title=r["title"],
                source=r["source"],
                content="",
                metadata=json.loads(r["metadata"]),
            )
            for r in rows
        ]

    def delete_document(self, doc_id: str) -> int:
        """Delete a document and its chunks; returns chunks removed."""
        cur = self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        removed = cur.rowcount
        self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self._conn.commit()
        return removed

    # -- chunks ------------------------------------------------------------------
    def upsert_chunks(self, doc_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        for chunk, vector in zip(chunks, embeddings):
            self._conn.execute(
                "INSERT OR REPLACE INTO chunks (id, doc_id, text, metadata, embedding) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    chunk.id,
                    doc_id,
                    chunk.text,
                    json.dumps(chunk.metadata),
                    json.dumps(vector),
                ),
            )
        self._conn.commit()

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            return None
        return Chunk(
            id=row["id"],
            doc_id=row["doc_id"],
            text=row["text"],
            metadata=json.loads(row["metadata"]),
        )

    def list_chunks(self) -> list[Chunk]:
        rows = self._conn.execute("SELECT * FROM chunks ORDER BY id").fetchall()
        return [
            Chunk(
                id=r["id"],
                doc_id=r["doc_id"],
                text=r["text"],
                metadata=json.loads(r["metadata"]),
            )
            for r in rows
        ]

    def list_chunks_with_embeddings(self) -> list[tuple[Chunk, list[float]]]:
        rows = self._conn.execute("SELECT * FROM chunks ORDER BY id").fetchall()
        out = []
        for r in rows:
            chunk = Chunk(
                id=r["id"],
                doc_id=r["doc_id"],
                text=r["text"],
                metadata=json.loads(r["metadata"]),
            )
            out.append((chunk, json.loads(r["embedding"])))
        return out

    def count_chunks(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"])

    def close(self) -> None:
        self._conn.close()
