"""Chunking: split documents into retrievable pieces.

Two strategies:

``fixed``
    Sliding-window chunking by characters with a configurable size and
    overlap. Splits prefer sentence boundaries, then paragraph breaks,
    then hard cuts — so chunks stay readable instead of splitting
    mid-sentence whenever possible.

``semantic``
    Sentence-level chunking: sentences are embedded, and a new chunk
    starts when the cosine similarity between neighbouring sentences
    drops below a threshold (i.e. the topic shifts). Needs an embedding
    model; without one it falls back to paragraph-based splitting.
"""

from __future__ import annotations

import re

from eki.models import Chunk, Document

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_PAGE_MARKER = re.compile(r"\[Page (\d+)\]")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping the punctuation attached."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def _page_of(text: str, offset: int) -> int | None:
    """Find the most recent ``[Page N]`` marker before ``offset``."""
    page = None
    for match in _PAGE_MARKER.finditer(text[:offset]):
        page = int(match.group(1))
    return page


def chunk_fixed(
    doc: Document,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Sliding-window chunking with sentence-aware boundaries.

    The window advances by ``chunk_size - chunk_overlap`` characters,
    but the cut point is pulled back to the last sentence boundary
    inside the window so chunks end on a complete sentence.
    """
    text = doc.content
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            # Pull the cut back to the last sentence end inside the window.
            window = text[start:end]
            last_break = max(
                window.rfind(". "),
                window.rfind("! "),
                window.rfind("? "),
                window.rfind("\n\n"),
            )
            if last_break > chunk_size // 2:  # only if it leaves a decent chunk
                end = start + last_break + 1
            else:
                # No sentence break: fall back to the last word boundary
                # so we never cut mid-word.
                last_space = window.rfind(" ")
                if last_space > chunk_size // 2:
                    end = start + last_space
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    id=f"{doc.id}-c{index}",
                    doc_id=doc.id,
                    text=piece,
                    metadata={
                        "title": doc.title,
                        **doc.metadata,
                        "chunk_index": index,
                        "page": _page_of(text, start),
                        "strategy": "fixed",
                    },
                )
            )
            index += 1
        if end >= text_len:
            break
        # Advance by (chunk_size - overlap), snapping the new start to a
        # word boundary so the next chunk never starts mid-word.
        start = max(end - chunk_overlap, start + 1)
        while start < text_len and not text[start].isspace():
            start += 1
        while start < text_len and text[start].isspace():
            start += 1

    return chunks


def chunk_semantic(
    doc: Document,
    embed_fn=None,
    similarity_threshold: float = 0.55,
    max_chunk_chars: int = 1200,
    min_chunk_chars: int = 100,
) -> list[Chunk]:
    """Semantic chunking: split where the topic shifts.

    Sentences are embedded with ``embed_fn`` (a callable taking a list
    of strings and returning a list of vectors). A new chunk starts when
    the cosine similarity between consecutive sentence embeddings falls
    below ``similarity_threshold``, or when the chunk would exceed
    ``max_chunk_chars``. Without an embedding function the splitter
    falls back to paragraph boundaries.
    """
    sentences = split_sentences(doc.content)
    if not sentences:
        return []

    # Group sentences into topic-coherent blocks.
    blocks: list[list[str]] = []
    current: list[str] = [sentences[0]]
    current_len = len(sentences[0])

    if embed_fn is not None:
        import numpy as np

        embeddings = np.asarray(embed_fn(sentences), dtype=float)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = embeddings / norms
        for i in range(1, len(sentences)):
            sim = float(unit[i - 1] @ unit[i])
            would_overflow = current_len + len(sentences[i]) > max_chunk_chars
            if sim < similarity_threshold or would_overflow:
                blocks.append(current)
                current = [sentences[i]]
                current_len = len(sentences[i])
            else:
                current.append(sentences[i])
                current_len += len(sentences[i])
    else:
        # No embedder: fall back to paragraph-ish grouping by length.
        for sentence in sentences[1:]:
            if current_len + len(sentence) > max_chunk_chars:
                blocks.append(current)
                current = [sentence]
                current_len = len(sentence)
            else:
                current.append(sentence)
                current_len += len(sentence)
    if current:
        blocks.append(current)

    chunks: list[Chunk] = []
    offset = 0
    for index, block in enumerate(blocks):
        piece = " ".join(block).strip()
        if len(piece) < min_chunk_chars and len(blocks) > 1:
            # Tiny tail block: merge into the previous chunk.
            if chunks:
                chunks[-1].text += " " + piece
                continue
        chunks.append(
            Chunk(
                id=f"{doc.id}-c{index}",
                doc_id=doc.id,
                text=piece,
                metadata={
                    "title": doc.title,
                    **doc.metadata,
                    "chunk_index": index,
                    "page": _page_of(doc.content, offset),
                    "strategy": "semantic",
                },
            )
        )
        offset += len(piece)
    return chunks


def chunk_document(
    doc: Document,
    strategy: str = "fixed",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    embed_fn=None,
) -> list[Chunk]:
    """Chunk a document with the requested strategy."""
    if strategy == "semantic":
        return chunk_semantic(doc, embed_fn=embed_fn)
    return chunk_fixed(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
