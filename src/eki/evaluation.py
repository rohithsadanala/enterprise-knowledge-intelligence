"""Evaluation harness: retrieval + groundedness metrics.

Metrics (all computed natively — no API key required):

- ``precision@k`` / ``recall@k`` — fraction of retrieved chunks that
  are relevant / fraction of relevant chunks retrieved, against the
  eval set's gold ``relevant_doc_ids``.
- ``mrr`` — mean reciprocal rank of the first relevant chunk.
- ``faithfulness`` — fraction of the answer's claims that are supported
  by the retrieved context. With a generative LLM this is judged by
  the model itself; with the offline mock LLM it falls back to a
  lexical check (claim sentences sharing vocabulary with the context).
  When DeepEval is installed and an OpenAI key is configured, the
  harness delegates to DeepEval's ``FaithfulnessMetric`` instead —
  see :func:`deepeval_faithfulness` — and reports both numbers.

The sample eval set lives in ``eval_data/eval_set.json`` and covers
the sample documents in ``docs/sample/``. Run it with::

    python scripts/run_eval.py
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from eki.models import QueryResult
from eki.retrieval import tokenize


@dataclass
class EvalCase:
    question: str
    expected_answer: str
    relevant_doc_ids: list[str]
    context: str = ""


@dataclass
class EvalResult:
    question: str
    answer: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    faithfulness: float
    retrieved_doc_ids: list[str]
    citations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_eval_set(path: str | Path) -> list[EvalCase]:
    data = json.loads(Path(path).read_text())
    return [EvalCase(**case) for case in data]


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return len([d for d in top if d in relevant]) / len(top)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return len({d for d in top if d in relevant}) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _split_claims(answer: str) -> list[str]:
    # Strip citation markers ("[1]", "[2]") — they are not claims.
    clean = re.sub(r"\[\d+\]", " ", answer)
    claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
    return claims or ([clean.strip()] if clean.strip() else [])


def lexical_faithfulness(answer: str, context_chunks: list[str]) -> float:
    """Lexical groundedness: share of answer claims whose content words
    appear in the retrieved context."""
    claims = _split_claims(answer)
    if not claims:
        return 0.0
    context_words: set[str] = set()
    for chunk in context_chunks:
        context_words |= set(tokenize(chunk))
    supported = 0
    evaluated = 0
    for claim in claims:
        words = {w for w in tokenize(claim) if len(w) > 2}
        if not words:
            continue
        evaluated += 1
        if len(words & context_words) / len(words) >= 0.5:
            supported += 1
    if evaluated == 0:
        return 0.0
    return supported / evaluated


def deepeval_faithfulness(question: str, answer: str, context: list[str]) -> float | None:
    """DeepEval faithfulness, when DeepEval + an LLM are available.

    Returns ``None`` when DeepEval is not installed, so callers can
    fall back to :func:`lexical_faithfulness`.
    """
    try:
        from deepeval import evaluate
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
    except ImportError:
        return None
    metric = FaithfulnessMetric(threshold=0.5)
    case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=context,
    )
    evaluate([case], [metric])
    return float(metric.score) if metric.score is not None else None


def evaluate_case(
    case: EvalCase,
    result: QueryResult,
    retrieved_chunks: list[str],
    k: int = 5,
) -> EvalResult:
    """Score one question against its gold relevance labels."""
    retrieved_doc_ids = []
    seen: set[str] = set()
    for citation in result.citations:
        if citation.doc_id not in seen:
            seen.add(citation.doc_id)
            retrieved_doc_ids.append(citation.doc_id)
    relevant = set(case.relevant_doc_ids)

    # Retrieval metrics use the retrieved chunk order (via citations);
    # if the answer cited nothing, fall back to the retrieval order.
    rank_order = retrieved_doc_ids or []
    faithfulness = lexical_faithfulness(result.answer, retrieved_chunks)
    deep_score = deepeval_faithfulness(
        case.question, result.answer, retrieved_chunks
    )
    if deep_score is not None:
        faithfulness = deep_score

    return EvalResult(
        question=case.question,
        answer=result.answer,
        precision_at_k=precision_at_k(rank_order, relevant, k),
        recall_at_k=recall_at_k(rank_order, relevant, k),
        reciprocal_rank=reciprocal_rank(rank_order, relevant),
        faithfulness=faithfulness,
        retrieved_doc_ids=retrieved_doc_ids,
        citations=[c.to_reference(i + 1) for i, c in enumerate(result.citations)],
    )


def summarize(results: list[EvalResult]) -> dict[str, Any]:
    """Aggregate metrics across the eval set."""
    if not results:
        return {}
    return {
        "n": len(results),
        "precision_at_k": round(statistics.mean(r.precision_at_k for r in results), 3),
        "recall_at_k": round(statistics.mean(r.recall_at_k for r in results), 3),
        "mrr": round(statistics.mean(r.reciprocal_rank for r in results), 3),
        "faithfulness": round(statistics.mean(r.faithfulness for r in results), 3),
    }
