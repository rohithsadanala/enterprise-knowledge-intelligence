"""Answer generation: a LangChain chain with citations.

The chain is built from LangChain primitives:

- :class:`langchain_core.prompts.PromptTemplate` — the grounded
  answering prompt, which instructs the model to answer **only** from
  the supplied context and to cite every claim as ``[n]``.
- A ``RunnableLambda`` wrapping the configured LLM (OpenAI, Hugging
  Face or the offline mock).
- :class:`langchain_core.output_parsers.StrOutputParser` — plain text.

The context block is numbered (``[1]``, ``[2]``, ...) so the citation
markers in the answer map directly onto the ``Citation`` objects the
pipeline returns. After generation, :func:`extract_citations` parses
the markers out of the answer and :func:`build_citations` resolves
them to chunks — so an answer always carries its sources, even if the
model forgets to cite (in that case the top retrieved chunks are
attached as sources).
"""

from __future__ import annotations

import re

from eki.models import Chunk, Citation

ANSWER_PROMPT = """You are an enterprise knowledge assistant. Answer the question using ONLY the context below. Every factual claim in your answer must be followed by a citation marker like [1] or [2] referring to the numbered source it came from. If the context does not contain the answer, say so clearly and do not invent information.

Context:
{context}

Question: {question}

Answer (with citations):"""

_CITATION_RE = re.compile(r"\[(\d+)\]")


def format_context(chunks: list[Chunk]) -> str:
    """Render retrieved chunks as a numbered context block.

    Each source is a block with a ``Title:`` line and a ``---``
    separator, so extractive readers (and the mock LLM) can tell the
    source label apart from the chunk text.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.metadata.get("title") or chunk.doc_id
        blocks.append(f"[{i}]\nTitle: {title}\n---\n{chunk.text}")
    return "\n\n".join(blocks)


def build_answer_chain(llm):
    """Build the LangChain answer chain for the given LLM.

    Falls back to a plain prompt-format + ``llm.generate`` call when
    LangChain core is unavailable.
    """
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import PromptTemplate
        from langchain_core.runnables import RunnableLambda

        prompt = PromptTemplate.from_template(ANSWER_PROMPT)
        chain = prompt | RunnableLambda(lambda inputs: llm.generate(
            ANSWER_PROMPT.format(context=inputs["context"], question=inputs["question"])
        )) | StrOutputParser()
        return chain
    except Exception:
        return None


def generate_answer(llm, question: str, chunks: list[Chunk]) -> str:
    """Generate a cited answer for a question given retrieved chunks."""
    if not chunks:
        return "I don't have enough information in the knowledge base to answer that question."
    context = format_context(chunks)
    chain = build_answer_chain(llm)
    if chain is not None:
        try:
            return str(chain.invoke({"context": context, "question": question})).strip()
        except Exception:
            pass
    # Fallback: direct prompt + generate.
    return llm.generate(
        ANSWER_PROMPT.format(context=context, question=question)
    ).strip()


def extract_citations(answer: str) -> list[int]:
    """Return the sorted, deduplicated source numbers cited in an answer."""
    return sorted({int(n) for n in _CITATION_RE.findall(answer)})


def build_citations(chunks: list[Chunk], answer: str) -> list[Citation]:
    """Resolve an answer's ``[n]`` markers to :class:`Citation` objects.

    Only chunks actually cited by the answer are returned, in citation
    order. Callers that want *all* retrieved chunks as sources should
    use :func:`all_citations` instead.
    """
    cited = extract_citations(answer)
    out: list[Citation] = []
    for n in cited:
        if 1 <= n <= len(chunks):
            chunk = chunks[n - 1]
            out.append(_citation_for(chunk))
    return out


def all_citations(chunks: list[Chunk]) -> list[Citation]:
    """One :class:`Citation` per retrieved chunk, in retrieval order."""
    return [_citation_for(chunk) for chunk in chunks]


def _citation_for(chunk: Chunk) -> Citation:
    title = (
        chunk.metadata.get("title")
        or chunk.metadata.get("source")
        or chunk.doc_id
    )
    return Citation(
        doc_id=chunk.doc_id,
        doc_title=str(title),
        chunk_id=chunk.id,
        score=chunk.score,
        snippet=chunk.text[:300],
        page=chunk.metadata.get("page"),
    )
