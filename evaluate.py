"""
Evaluation suite: benchmark ColPali retrieval on diverse query types
covering text, tables, charts, and figures in NASA technical reports.
"""

from __future__ import annotations

import json
import time
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

# Benchmark queries spanning multiple modalities
BENCHMARK_QUERIES = [
    # Text-based
    {"query": "What are the main objectives of this research?", "type": "text", "expected_modality": "text"},
    {"query": "What conclusions were drawn from the experiment?", "type": "text", "expected_modality": "text"},
    {"query": "Describe the methodology used in this study.", "type": "text", "expected_modality": "text"},

    # Table-based
    {"query": "What are the performance values in the results table?", "type": "table", "expected_modality": "table"},
    {"query": "Show the numerical comparison of different configurations.", "type": "table", "expected_modality": "table"},
    {"query": "What are the material properties listed?", "type": "table", "expected_modality": "table"},

    # Figure/chart-based
    {"query": "What does the pressure vs velocity graph show?", "type": "figure", "expected_modality": "figure"},
    {"query": "What trend is visible in the performance chart?", "type": "figure", "expected_modality": "figure"},
    {"query": "Describe the system architecture diagram.", "type": "figure", "expected_modality": "figure"},

    # Cross-modal
    {"query": "What is the maximum thrust efficiency achieved and at what temperature?", "type": "cross-modal", "expected_modality": "mixed"},
    {"query": "Compare the experimental results shown in the figure with the data in the table.", "type": "cross-modal", "expected_modality": "mixed"},
]


def evaluate_retrieval(
    retriever,
    queries: List[Dict],
    top_k: int = 5,
) -> Dict[str, Any]:
    results = []
    latencies = []

    for q in queries:
        start = time.time()
        hits = retriever.retrieve(q["query"], top_k=top_k)
        elapsed = time.time() - start
        latencies.append(elapsed)

        results.append({
            "query": q["query"],
            "type": q["type"],
            "expected_modality": q["expected_modality"],
            "top_result": hits[0] if hits else None,
            "num_results": len(hits),
            "latency_s": round(elapsed, 3),
            "top_score": hits[0]["score"] if hits else 0.0,
            "retrieved_pages": [
                {"doc": h["doc_name"], "page": h["page_num"], "score": round(h["score"], 4)}
                for h in hits[:3]
            ],
        })

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    avg_top_score = sum(r["top_score"] for r in results) / len(results) if results else 0

    return {
        "summary": {
            "total_queries": len(queries),
            "avg_latency_s": round(avg_latency, 3),
            "avg_top_score": round(avg_top_score, 4),
            "queries_by_type": {
                t: len([r for r in results if r["type"] == t])
                for t in set(q["type"] for q in queries)
            },
        },
        "results": results,
    }


def print_report(eval_output: Dict[str, Any]):
    summary = eval_output["summary"]
    print("\n" + "=" * 60)
    print("EVALUATION REPORT — ColPali Retrieval Benchmark")
    print("=" * 60)
    print(f"Total queries:      {summary['total_queries']}")
    print(f"Avg latency:        {summary['avg_latency_s']:.3f}s")
    print(f"Avg top-1 score:    {summary['avg_top_score']:.4f}")
    print(f"Query types:        {summary['queries_by_type']}")
    print()

    for r in eval_output["results"]:
        print(f"[{r['type'].upper():12s}] {r['query'][:55]:<55} | score={r['top_score']:.4f} | {r['latency_s']:.2f}s")
        if r["retrieved_pages"]:
            for hit in r["retrieved_pages"]:
                print(f"    → {hit['doc'][:40]} p.{hit['page']} ({hit['score']:.4f})")
    print()


if __name__ == "__main__":
    from src.retrieval import build_retriever
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading retriever on {device}...")
    retriever = build_retriever(qdrant_path="data/qdrant_db", device=device)

    print(f"Running {len(BENCHMARK_QUERIES)} benchmark queries...")
    eval_output = evaluate_retrieval(retriever, BENCHMARK_QUERIES, top_k=5)

    print_report(eval_output)

    # Save results
    out_path = "data/eval_results.json"
    with open(out_path, "w") as f:
        json.dump(eval_output, f, indent=2)
    print(f"Results saved to {out_path}")
