"""Command-line interface for Enterprise Knowledge Intelligence.

Usage::

    # Ingest a folder of documents
    python -m eki.cli ingest docs/sample

    # Ask a question
    python -m eki.cli ask "What is the remote work policy?"

    # Ask with metadata filtering
    python -m eki.cli ask "What is the PTO policy?" --filter file_type=.pdf

    # Show index stats
    python -m eki.cli stats

    # Start the API server
    python -m eki.cli serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eki.pipeline import build_pipeline


def _print_result(result) -> None:
    print(f"\nAnswer ({result.latency_ms:.0f} ms, {result.retrieved_count} chunks retrieved):")
    print("-" * 70)
    print(result.answer)
    print("-" * 70)
    print("Sources:")
    for i, citation in enumerate(result.citations, start=1):
        print(f"  {citation.to_reference(i)}  (score={citation.score})")
    print(f"\nQuery plan: {json.dumps(result.query_plan)}")


def cmd_ingest(args) -> int:
    pipeline = build_pipeline()
    path = Path(args.path)
    if path.is_dir():
        results = pipeline.ingest_directory(path)
        total_chunks = sum(r["chunks"] for r in results)
        print(f"Ingested {len(results)} document(s), {total_chunks} chunks:")
        for r in results:
            print(f"  - {r['title']}: {r['chunks']} chunks (id={r['doc_id']})")
    elif path.is_file():
        result = pipeline.ingest_file(path)
        print(f"Ingested {result['title']}: {result['chunks']} chunks (id={result['doc_id']})")
    else:
        print(f"error: no such file or directory: {path}", file=sys.stderr)
        return 2
    print(f"Index now holds {pipeline.stats()['chunks']} chunks.")
    return 0


def cmd_ask(args) -> int:
    pipeline = build_pipeline()
    filters = None
    if args.filter:
        filters = {}
        for item in args.filter:
            if "=" not in item:
                print(f"error: --filter expects key=value, got {item!r}", file=sys.stderr)
                return 2
            key, value = item.split("=", 1)
            filters[key] = value
    result = pipeline.query(args.question, top_k=args.top_k, filters=filters)
    if args.json:
        print(json.dumps({
            "question": result.question,
            "answer": result.answer,
            "citations": [vars(c) for c in result.citations],
            "query_plan": result.query_plan,
            "latency_ms": round(result.latency_ms, 1),
            "retrieved_count": result.retrieved_count,
        }, indent=2))
    else:
        _print_result(result)
    return 0


def cmd_stats(args) -> int:
    pipeline = build_pipeline()
    print(json.dumps(pipeline.stats(), indent=2))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run(
        "eki.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eki",
        description="Enterprise Knowledge Intelligence — advanced RAG over your documents.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest a file or a folder of documents.")
    p_ingest.add_argument("path", help="File or directory to ingest.")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a question against the knowledge base.")
    p_ask.add_argument("question", help="The question to answer.")
    p_ask.add_argument("--top-k", type=int, default=None, help="Retrieval depth.")
    p_ask.add_argument("--filter", action="append", default=None,
                       help="Metadata filter as key=value (repeatable).")
    p_ask.add_argument("--json", action="store_true", help="Print the raw JSON result.")
    p_ask.set_defaults(func=cmd_ask)

    p_stats = sub.add_parser("stats", help="Show index statistics.")
    p_stats.set_defaults(func=cmd_stats)

    p_serve = sub.add_parser("serve", help="Start the FastAPI server.")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
