"""
Decomposition Evaluation Script

Evaluates query decomposition + Neo4j vector search on the legal QA dataset.

Usage:
    uv run scripts/eval_rag_decomposition.py \
        --input qa_dataset/QA_NLP.csv \
        --output eval_results_decomposition \
        --uri "neo4j+ssc://host:7687" \
        --user neo4j \
        --password "..." \
        --database neo4j
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


# ─────────────────────────────────────────────────────────────────────────────
# Relevance & Metrics
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
    parser = argparse.ArgumentParser(description="Evaluate RAG with query decomposition")
    parser.add_argument("--input", required=True, help="Path to QA_NLP.csv")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument(
        "--uri",
        default=os.environ.get("NEO4J_URI", "neo4j+ssc://localhost:7687"),
        help="Neo4j URI (or NEO4J_URI env var)",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j user (or NEO4J_USER env var)",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("NEO4J_PASSWORD", ""),
        help="Neo4j password (or NEO4J_PASSWORD env var)",
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("NEO4J_DATABASE", "neo4j"),
        help="Neo4j database (or NEO4J_DATABASE env var)",
    )
    parser.add_argument(
        "--agg",
        default="rrf",
        choices=["rrf", "borda", "max"],
        help="Aggregation strategy to merge sub-query results (default: rrf)",
    )
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

    print("Loading DB Embedder...")
    embedder = Neo4jEmbedder(args.uri, args.user, args.password, args.database)
    
    print("Loading Query Decomposer...")
    from legal_scraper.query_parser import QueryDecomposer
    decomposer = QueryDecomposer()

    print("Loading Reranker...")
    reranker = VietnameseReranker(device="cpu")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    row_path = out_dir / "row_results_decomposition.csv"
    summary_path = out_dir / "metrics_summary_decomposition.csv"

    # --- Resume capability ---
    processed_ids = set()
    if row_path.exists():
        try:
            existing_df = pd.read_csv(row_path)
            processed_ids = set(existing_df["id"].tolist())
            print(f"Found existing results. Resuming from {len(processed_ids)} already processed questions.")
            row_records = existing_df.to_dict("records")
        except Exception as e:
            print(f"Could not read existing results: {e}")
            row_records = []
    else:
        row_records = []
    metric_names = ["recall@1", "recall@3", "recall@5", "recall@7", "recall@10",
                    "precision@1", "precision@3", "mrr"]
    skipped = 0

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

        # Explicitly call decompose to check for LLM backend failure before pipeline
        try:
            sub_queries = decomposer.decompose(question)
        except Exception as e:
            print(f"\n[CRITICAL ERROR] LLM Backend API failed at row {idx} (ID: {row_id}).")
            print(f"Error details: {e}")
            print("Stopping evaluation to prevent silent fallback to standard RAG.")
            print("Please RESTART your Colab/API server to clear GPU RAM, then re-run this script to automatically resume from this row.")
            sys.exit(1)
            
        # Fallback to original query if no sub-queries generated
        if not sub_queries:
            sub_queries = [{"query": question}]

        # Execute search directly using the generated sub-queries to avoid double LLM calls
        raw_results = embedder.multi_search(sub_queries, k=60) # k*2 for aggregation
        
        from legal_scraper.retrieval import aggregate_search_results, fetch_context_for_results
        search_results = aggregate_search_results(raw_results, strategy=args.agg)[:30]
        
        # Cross-Encoder Reranking
        if search_results:
            context_map = fetch_context_for_results(embedder, search_results, include_hierarchy=True)
            docs = [context_map.get((r.uid, r.label), "") for r in search_results]
            reranked_indices = reranker.rerank(question, docs, top_k=len(docs), batch_size=4)
            retrieved_uids = [search_results[idx].uid for idx, _ in reranked_indices]
        else:
            retrieved_uids = []

        # Compute Metrics
        row_metrics = compute_row_metrics(retrieved_uids, references)

        new_record = {
            "id": row_id,
            "question": question,
            "retrieved_uids": ";".join(retrieved_uids),
            "references": ";".join(references),
            **row_metrics,
        }
        row_records.append(new_record)
        processed_ids.add(row_id)

        # Save immediately to prevent data loss if backend crashes
        pd.DataFrame([new_record]).to_csv(row_path, mode='a', header=not row_path.exists(), index=False)

        if len(processed_ids) % 10 == 0:
            print(f"  Processed {len(processed_ids)}/{len(df)} questions ...")
            
        # Give the LLM backend time to run Garbage Collection / clear VRAM between requests
        time.sleep(3)

    print(f"\nDone — {len(row_records)} rows evaluated, {skipped} skipped.")

    # Update final summary
    if row_records:
        # Re-save the full sorted file just to be clean
        row_df = pd.DataFrame(row_records)
        row_df.to_csv(row_path, index=False)
        
        summary = {m: sum(r[m] for r in row_records) / len(row_records) for m in metric_names}
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary → {summary_path}")

        print("\n=== Averaged Metrics (DECOMPOSITION) ===")
        for m, v in summary.items():
            print(f"  {m:<15}: {v:.4f}")

    embedder.close()


if __name__ == "__main__":
    main()
