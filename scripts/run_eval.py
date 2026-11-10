"""Run the evaluation harness over the sample eval set.

Ingests ``docs/sample/`` (if the index is empty), runs every question
in ``eval_data/eval_set.json`` through the pipeline, and prints a
per-question table plus aggregate metrics. Results are written to
``eval_data/eval_results.json``.

Usage::

    python scripts/run_eval.py [--eval-set eval_data/eval_set.json] [--k 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eki.evaluation import evaluate_case, load_eval_set, summarize  # noqa: E402
from eki.pipeline import build_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the EKI evaluation harness.")
    parser.add_argument("--eval-set", default="eval_data/eval_set.json")
    parser.add_argument("--k", type=int, default=5, help="k for precision/recall@k")
    parser.add_argument("--sample-docs", default="docs/sample")
    args = parser.parse_args()

    pipeline = build_pipeline()
    if pipeline.stats()["documents"] == 0:
        print(f"Index is empty — ingesting {args.sample_docs} ...")
        pipeline.ingest_directory(args.sample_docs)

    cases = load_eval_set(args.eval_set)
    print(f"Evaluating {len(cases)} cases (k={args.k}) ...\n")

    # The eval set names relevant docs by file stem ("remote-work-policy");
    # resolve those to the real indexed doc ids (stem + content hash).
    title_to_id = {d.title: d.id for d in pipeline.list_documents()}
    for case in cases:
        case.relevant_doc_ids = [
            title_to_id.get(stem, stem) for stem in case.relevant_doc_ids
        ]

    results = []
    for case in cases:
        result = pipeline.query(case.question)
        # The retrieved chunk texts for the faithfulness check come from
        # the pipeline's chunk store, in citation order.
        chunk_texts = []
        for citation in result.citations:
            chunk = pipeline.chunks.get(citation.chunk_id)
            if chunk is not None:
                chunk_texts.append(chunk.text)
        evaluated = evaluate_case(case, result, chunk_texts, k=args.k)
        results.append(evaluated)
        print(f"Q: {case.question}")
        print(f"   A: {evaluated.answer[:160]}")
        print(
            f"   P@{args.k}={evaluated.precision_at_k:.2f}  "
            f"R@{args.k}={evaluated.recall_at_k:.2f}  "
            f"MRR={evaluated.reciprocal_rank:.2f}  "
            f"faithfulness={evaluated.faithfulness:.2f}"
        )

    summary = summarize(results)
    print("\nAggregate:", json.dumps(summary, indent=2))

    out_path = Path(args.eval_set).parent / "eval_results.json"
    out_path.write_text(json.dumps(
        {"summary": summary, "results": [r.as_dict() for r in results]}, indent=2
    ))
    print(f"Results written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
