"""
V3 Hybrid Pipeline Evaluation Script

Evaluates the full v3 pipeline on the legal QA dataset:
  - Hierarchical routing (intent → complexity)
  - Query rewrite (simple) / decompose (complex)
  - Hybrid search (vector + BM25 keyword)
  - Cross-encoder reranking with ORIGINAL question

Outputs results to eval_results_v3/ and optionally compares with v2.

Usage:
    uv run scripts/eval_rag_v3_hybrid.py \
        --input qa_dataset/QA_NLP.csv \
        --output eval_results_v3 \
        --compare eval_results_v2/metrics_summary_decomposition.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from legal_scraper.embedder import Neo4jEmbedder
from legal_scraper.reranker import VietnameseReranker
from legal_scraper.retrieval import aggregate_search_results, fetch_context_for_results


# ─────────────────────────────────────────────────────────────────────────────
# Relevance & Metrics (same as v2 for fair comparison)
# ─────────────────────────────────────────────────────────────────────────────

def is_relevant(retrieved_uid: str, reference: str) -> bool:
    """Return True if `retrieved_uid` shares a prefix with `reference`."""
    return retrieved_uid.startswith(reference)


def recall_at_k(relevant_in_top_k: int, total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    return relevant_in_top_k / total_relevant


def precision_at_k(relevant_in_top_k: int, k: int) -> float:
    if k == 0:
        return 0.0
    return relevant_in_top_k / k


def mrr(retrieved_uids: list[str], references: list[str]) -> float:
    """Mean Reciprocal Rank: 1 / rank of first relevant item (0 if none)."""
    for i, uid in enumerate(retrieved_uids, start=1):
        for ref in references:
            if is_relevant(uid, ref):
                return 1.0 / i
    return 0.0


def compute_row_metrics(retrieved_uids: list[str], references: list[str]):
    total_relevant = len(references)
    metrics = {}

    for k in [1, 3, 5, 7, 10]:
        top_k = retrieved_uids[:k]
        found_refs = {ref for uid in top_k for ref in references if is_relevant(uid, ref)}
        rel_in_k = len(found_refs)
        metrics[f"recall@{k}"] = recall_at_k(rel_in_k, total_relevant)
        if k in [1, 3]:
            metrics[f"precision@{k}"] = precision_at_k(rel_in_k, k)

    metrics["mrr"] = mrr(retrieved_uids, references)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate V3 Hybrid RAG Pipeline")
    parser.add_argument("--input", required=True, help="Path to QA_NLP.csv")
    parser.add_argument("--output", default="eval_results_v3", help="Output directory for results")
    parser.add_argument(
        "--uri",
        default=os.environ.get("NEO4J_URI", "neo4j+ssc://localhost:7687"),
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("NEO4J_PASSWORD", ""),
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("NEO4J_DATABASE", "neo4j"),
    )
    parser.add_argument(
        "--agg",
        default="rrf",
        choices=["rrf", "borda", "max"],
        help="Aggregation strategy (default: rrf)",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Path to v2 metrics_summary CSV for side-by-side comparison",
    )
    parser.add_argument(
        "--hybrid",
        dest="hybrid",
        action="store_true",
        help="Enable hybrid search (vector + BM25)",
    )
    parser.add_argument(
        "--no-hybrid",
        dest="hybrid",
        action="store_false",
        help="Disable hybrid search (vector only, for ablation)",
    )
    parser.set_defaults(hybrid=False)
    return parser.parse_args()


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"question", "reference"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def parse_reference(ref_str: str) -> list[str]:
    if not isinstance(ref_str, str) or not ref_str.strip():
        return []
    return [r.strip() for r in ref_str.split(",") if r.strip()]


def main():
    args = parse_args()
    df = load_dataset(args.input)
    print(f"Loaded {len(df)} questions from {args.input}")

    # --- Load components ---
    print("Loading DB Embedder...")
    embedder = Neo4jEmbedder(args.uri, args.user, args.password, args.database)

    print("Loading Query Decomposer...")
    from legal_scraper.query_parser import QueryDecomposer
    decomposer = QueryDecomposer()

    print("Loading Reranker...")
    reranker = VietnameseReranker(device="cpu")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = "hybrid" if args.hybrid else "vector_only"
    row_path = out_dir / f"row_results_{tag}.csv"
    summary_path = out_dir / f"metrics_summary_{tag}.csv"

    # --- Resume capability ---
    processed_ids = set()
    if row_path.exists():
        try:
            existing_df = pd.read_csv(row_path)
            processed_ids = set(existing_df["id"].tolist())
            print(f"Resuming from {len(processed_ids)} already processed questions.")
            row_records = existing_df.to_dict("records")
        except Exception as e:
            print(f"Could not read existing results: {e}")
            row_records = []
    else:
        row_records = []

    metric_names = ["recall@1", "recall@3", "recall@5", "recall@7", "recall@10",
                    "precision@1", "precision@3", "mrr"]
    skipped = 0
    fallback_count = 0

    for idx, row in df.iterrows():
        question = row["question"]
        row_id = row.get("id", idx + 1)

        if row_id in processed_ids:
            continue

        if not isinstance(question, str) or not question.strip():
            print(f"  Row {idx}: empty question — skipped")
            skipped += 1
            continue

        references = parse_reference(str(row.get("reference", "")))
        if not references:
            print(f"  Row {idx}: no references — skipped")
            skipped += 1
            continue

        # --- Step 1: Decompose ---
        print(f"\n[{row_id}/{len(df)}] Q: {question[:80]}{'...' if len(question) > 80 else ''}")

        try:
            sub_queries = decomposer.decompose(question)
        except Exception as e:
            print(f"\n[CRITICAL ERROR] LLM Backend API failed at row {idx} (ID: {row_id}).")
            print(f"Error details: {e}")
            print("Stopping evaluation. Re-run to resume.")
            sys.exit(1)

        if not sub_queries:
            sub_queries = [{"query": question}]
            fallback_count += 1

        # Always include original query as fallback for BM25 keyword coverage
        sub_queries.append({"query": question})

        # Build rerank query from decomposed sub-queries (exclude original)
        rerank_query = " ".join([sq["query"] for sq in sub_queries[:-1]])
        print(f"  Sub-queries: {len(sub_queries)} (incl. original)")
        for sq in sub_queries[:-1]:
            print(f"    → {sq['query'][:80]}")

        # --- Step 2: Hybrid or Vector-only search ---
        raw_results = embedder.multi_search(sub_queries, k=60, hybrid=args.hybrid)
        search_results = aggregate_search_results(raw_results, strategy=args.agg)[:30]
        print(f"  Search: {len(search_results)} candidates (hybrid={args.hybrid})")

        # --- Step 3: Cross-encoder reranking with ORIGINAL question ---
        if search_results:
            context_map = fetch_context_for_results(embedder, search_results, include_hierarchy=True)
            docs = [context_map.get((r.uid, r.label), "") for r in search_results]
            reranked_indices = reranker.rerank(question, docs, top_k=len(docs), batch_size=4)
            retrieved_uids = [search_results[i].uid for i, _ in reranked_indices]
        else:
            retrieved_uids = []

        # --- Step 4: Compute metrics ---
        row_metrics = compute_row_metrics(retrieved_uids, references)
        hit = "✓" if row_metrics["recall@5"] > 0 else "✗"
        print(f"  {hit} R@1={row_metrics['recall@1']:.2f} R@3={row_metrics['recall@3']:.2f} R@5={row_metrics['recall@5']:.2f} MRR={row_metrics['mrr']:.2f}")

        new_record = {
            "id": row_id,
            "question": question,
            "num_subqueries": len(sub_queries),
            "decomposed_query": rerank_query,
            "retrieved_uids": ";".join(retrieved_uids),
            "references": ";".join(references),
            **row_metrics,
        }
        row_records.append(new_record)
        processed_ids.add(row_id)

        # Save incrementally
        pd.DataFrame([new_record]).to_csv(
            row_path, mode='a', header=not row_path.exists() or len(row_records) == 1,
            index=False
        )

        if len(processed_ids) % 10 == 0:
            print(f"  Processed {len(processed_ids)}/{len(df)} questions ...")

        # Give LLM backend time to clear VRAM
        time.sleep(3)

    print(f"\nDone — {len(row_records)} rows evaluated, {skipped} skipped, {fallback_count} fallbacks.")

    # --- Save final summary ---
    if row_records:
        row_df = pd.DataFrame(row_records)
        row_df.to_csv(row_path, index=False)

        summary = {m: sum(r[m] for r in row_records) / len(row_records) for m in metric_names}
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary → {summary_path}")

        print(f"\n=== V3 Metrics ({tag.upper()}) ===")
        for m, v in summary.items():
            print(f"  {m:<15}: {v:.4f}")

        # --- Comparison with V2 ---
        compare_path = args.compare
        if compare_path is None:
            # Auto-detect v2 summary
            v2_default = Path("eval_results_v2/metrics_summary_decomposition.csv")
            if v2_default.exists():
                compare_path = str(v2_default)

        if compare_path and Path(compare_path).exists():
            v2_df = pd.read_csv(compare_path)
            v2_summary = v2_df.iloc[0].to_dict()

            print(f"\n=== V2 vs V3 Comparison ===")
            print(f"  {'Metric':<15} {'V2':>8} {'V3':>8} {'Delta':>8}")
            print(f"  {'─' * 41}")
            for m in metric_names:
                v2_val = v2_summary.get(m, 0.0)
                v3_val = summary.get(m, 0.0)
                delta = v3_val - v2_val
                sign = "+" if delta >= 0 else ""
                print(f"  {m:<15} {v2_val:>8.4f} {v3_val:>8.4f} {sign}{delta:>7.4f}")

    embedder.close()


if __name__ == "__main__":
    main()
