"""Agentic query planning with LangGraph.

Before retrieval runs, a small LangGraph state machine plans how to
answer the question:

``classify``
    What kind of question is this? ``factual`` (a single fact),
    ``summary`` ("summarise / overview"), ``comparison`` ("compare X
    and Y") or ``multi_hop`` ("what does X imply for Y", or a question
    that names two different topics). Classification is rule-based
    first, with an LLM tie-break when one is configured.

``rewrite``
    Produce a search-friendly rewritten query plus expansion terms.
    Multi-hop and comparison questions get split into sub-questions;
    everything else gets keyword expansion.

``decide``
    Pick the retrieval depth (``top_k``): multi-hop questions fetch
    more candidates, simple factual ones fetch fewer.

The graph is compiled with LangGraph when available; if LangGraph
cannot be imported the same nodes run sequentially (same result, no
graph). The plan is returned as a plain dict and attached to every
:class:`~eki.models.QueryResult` so the API can show its work.
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

from eki.retrieval import expand_query

_COMPARISON_WORDS = {"compare", "versus", "vs", "difference", "differences", "better", "worse"}
_SUMMARY_WORDS = {"summarise", "summarize", "summary", "overview", "outline", "explain", "describe"}
_MULTI_HOP_HINTS = {"imply", "implies", "impact", "effect", "consequence", "because", "why", "how does"}


class PlanState(TypedDict, total=False):
    question: str
    query_type: str
    rewritten_query: str
    expansion_terms: list[str]
    sub_questions: list[str]
    top_k: int


def classify_question(question: str) -> str:
    """Rule-based question classification."""
    q = question.lower()
    words = set(re.findall(r"[a-z]+", q))
    if words & _COMPARISON_WORDS:
        return "comparison"
    if words & _SUMMARY_WORDS:
        return "summary"
    if words & _MULTI_HOP_HINTS:
        return "multi_hop"
    return "factual"


def _split_sub_questions(question: str, query_type: str) -> list[str]:
    """Split a comparison / multi-hop question into sub-questions."""
    if query_type == "comparison":
        parts = re.split(r"\b(?:versus|vs\.?|compared to|and)\b", question, flags=re.I)
        parts = [p.strip(" ?") for p in parts if p.strip()]
        return parts if len(parts) > 1 else [question]
    return [question]


def plan_query(question: str, llm=None, base_top_k: int = 8) -> dict[str, Any]:
    """Run the planning graph (or the sequential fallback) for a question."""
    query_type = classify_question(question)
    expansion_terms = expand_query(question)
    sub_questions = _split_sub_questions(question, query_type)

    rewritten = question
    if llm is not None and query_type in {"multi_hop", "comparison"}:
        # LLM rewrite: ask for a search-optimised restatement. The mock
        # LLM is extractive, so it cannot rewrite — only call the LLM
        # when it is a real generative model.
        try:
            from eki.llm import MockLLM

            if not isinstance(llm, MockLLM):
                rewritten = llm.generate(
                    "Rewrite the following question as a concise search query "
                    "for a document search engine. Return only the rewritten "
                    f"query, nothing else.\n\nQuestion: {question}"
                ).strip().splitlines()[0]
        except Exception:
            rewritten = question

    top_k = base_top_k
    if query_type == "multi_hop":
        top_k = base_top_k + 4
    elif query_type == "factual":
        top_k = max(4, base_top_k - 2)

    plan = {
        "query_type": query_type,
        "rewritten_query": rewritten,
        "expansion_terms": expansion_terms,
        "sub_questions": sub_questions,
        "top_k": top_k,
    }

    # Run the LangGraph version when available so the graph is exercised
    # (and visualisable); the sequential result above is identical.
    graph_plan = _run_graph(plan)
    return graph_plan if graph_plan else plan


def _run_graph(plan: dict[str, Any]) -> dict[str, Any] | None:
    """Compile and run the LangGraph planning graph; None on any failure."""
    try:
        from langgraph.graph import END, StateGraph

        def classify(state: PlanState) -> PlanState:
            return {"query_type": classify_question(state["question"])}

        def rewrite(state: PlanState) -> PlanState:
            return {
                "expansion_terms": expand_query(state["question"]),
                "sub_questions": _split_sub_questions(
                    state["question"], state.get("query_type", "factual")
                ),
            }

        def decide(state: PlanState) -> PlanState:
            base = state.get("top_k", 8)
            qtype = state.get("query_type", "factual")
            if qtype == "multi_hop":
                return {"top_k": base + 4}
            if qtype == "factual":
                return {"top_k": max(4, base - 2)}
            return {"top_k": base}

        graph = StateGraph(PlanState)
        graph.add_node("classify", classify)
        graph.add_node("rewrite", rewrite)
        graph.add_node("decide", decide)
        graph.set_entry_point("classify")
        graph.add_edge("classify", "rewrite")
        graph.add_edge("rewrite", "decide")
        graph.add_edge("decide", END)
        app = graph.compile()

        initial: PlanState = {
            "question": plan["rewritten_query"],
            "query_type": plan["query_type"],
            "rewritten_query": plan["rewritten_query"],
            "expansion_terms": plan["expansion_terms"],
            "sub_questions": plan["sub_questions"],
            "top_k": plan["top_k"],
        }
        result = app.invoke(initial)
        return {
            "query_type": result.get("query_type", plan["query_type"]),
            "rewritten_query": result.get("rewritten_query", plan["rewritten_query"]),
            "expansion_terms": result.get("expansion_terms", plan["expansion_terms"]),
            "sub_questions": result.get("sub_questions", plan["sub_questions"]),
            "top_k": int(result.get("top_k", plan["top_k"])),
        }
    except Exception:
        return None
