"""Tests for chunking (fixed + semantic fallback) and sentence splitting."""

from eki.chunking import chunk_document, chunk_fixed, chunk_semantic, split_sentences
from eki.models import Document


def _doc(text: str) -> Document:
    return Document(id="doc-1", title="Test", source="test.txt", content=text)


def test_split_sentences():
    text = "First sentence. Second sentence! Is this third? Yes, it is."
    sentences = split_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "First sentence."


def test_chunk_fixed_respects_size():
    text = " ".join(f"Sentence number {i} about the policy." for i in range(60))
    chunks = chunk_fixed(_doc(text), chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    for chunk in chunks:
        # Chunks may slightly exceed the window when a sentence is long,
        # but never by more than one sentence.
        assert len(chunk.text) <= 320
    # All chunks belong to the same document and are ordered.
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_chunk_fixed_overlap_covers_text():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_fixed(_doc(text), chunk_size=200, chunk_overlap=50)
    joined = " ".join(c.text for c in chunks)
    # With overlap, the middle of the text must appear in some chunk.
    assert "word250" in joined


def test_chunk_fixed_empty_document():
    assert chunk_fixed(_doc("   ")) == []


def test_chunk_semantic_falls_back_without_embedder():
    text = (
        "The remote policy allows three days at home. "
        "The PTO policy grants twenty days a year. "
        "Security requires multi-factor authentication everywhere. "
        "The stipend is five hundred dollars for home office setup."
    )
    chunks = chunk_semantic(_doc(text), embed_fn=None, max_chunk_chars=120)
    assert len(chunks) >= 2
    assert all(c.text for c in chunks)
    assert all(c.metadata["strategy"] == "semantic" for c in chunks)


def test_chunk_semantic_with_fake_embedder():
    from eki.embeddings import FakeEmbeddings

    embedder = FakeEmbeddings(dim=64)
    # Two clear topics: remote work vs. security.
    text = (
        "Remote work is allowed three days per week. "
        "Employees must be available during core hours. "
        "Passwords must be at least fourteen characters long. "
        "Multi-factor authentication is required on every system."
    )
    chunks = chunk_semantic(
        _doc(text), embed_fn=embedder.embed, max_chunk_chars=500
    )
    assert len(chunks) >= 1
    # The full text must survive chunking.
    assert "Remote work" in " ".join(c.text for c in chunks)
    assert "Multi-factor" in " ".join(c.text for c in chunks)


def test_chunk_document_dispatch():
    text = "Hello world. " * 50
    fixed = chunk_document(_doc(text), strategy="fixed", chunk_size=100, chunk_overlap=10)
    semantic = chunk_document(_doc(text), strategy="semantic")
    assert fixed and semantic
    assert fixed[0].metadata["strategy"] == "fixed"
    assert semantic[0].metadata["strategy"] == "semantic"
