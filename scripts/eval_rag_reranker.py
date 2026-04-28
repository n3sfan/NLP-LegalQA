"""
Reranker Evaluation Script

Evaluates Neo4j vector search + Cross-Encoder reranking on the legal QA dataset.

Usage:
    uv run scripts/eval_rag_reranker.py \
        --input qa_dataset/QA_NLP.csv \
        --output eval_results_reranker \
        --uri "neo4j+ssc://host:7687" \
        --user neo4j \
        --password "..." \
        --database neo4j
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from legal_scraper.embedder import Neo4jEmbedder
from legal_scraper.reranker import VietnameseReranker


# ─────────────────────────────────────────────────────────────────────────────
# Relevance & Metrics (Same as eval_rag.py)
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
    parser = argparse.ArgumentParser(description="Evaluate RAG with cross-encoder reranker")
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

    print("Loading DB Embedder & Reranker...")
    embedder = Neo4jEmbedder(args.uri, args.user, args.password, args.database)
    reranker = VietnameseReranker()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    row_records = []
    metric_names = ["recall@1", "recall@3", "recall@5", "recall@7", "recall@10",
                    "precision@1", "precision@3", "mrr"]
    skipped = 0

    for idx, row in df.iterrows():
        question = row["question"]
        if not isinstance(question, str) or not question.strip():
            print(f"  Row {idx}: empty question — skipped")
            skipped += 1
            continue

        references = parse_reference(str(row.get("reference", "")))
        if not references:
            print(f"  Row {idx}: no references — skipped")
            skipped += 1
            continue

        # 1. Base Vector Search
        search_results = embedder.search(labels=["Point", "Clause", "Article"], query=question, k=30)
        search_results = search_results[:30] # Limit to global top 30
        uids = [res.uid for res in search_results]

        # 2. Fetch context
        hierarchy_map = embedder.fetch_node_hierarchy(uids)

        documents = []
        valid_uids = []
        for res in search_results:
            if res.uid in hierarchy_map:
                documents.append(hierarchy_map[res.uid])
                valid_uids.append(res.uid)

        # 3. Cross-Encoder Reranking
        if documents:
            reranked_results = reranker.rerank(question, documents, top_k=30)
            retrieved_uids = [valid_uids[orig_idx] for orig_idx, _score in reranked_results]
        else:
            retrieved_uids = []

        # 4. Computer Metrics
        row_metrics = compute_row_metrics(retrieved_uids, references)

        row_records.append({
            "id": row.get("id", idx + 1),
            "question": question,
            "retrieved_uids": ";".join(retrieved_uids),
            "references": ";".join(references),
            **row_metrics,
        })

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)} questions ...")

    print(f"\nDone — {len(row_records)} rows evaluated, {skipped} skipped.")

    # Save outputs
    row_df = pd.DataFrame(row_records, columns=["id", "question", "retrieved_uids", "references"] + metric_names)
    row_path = out_dir / "row_results_reranker.csv"
    row_df.to_csv(row_path, index=False)
    print(f"Saved per-row results → {row_path}")

    if row_records:
        summary = {m: sum(r[m] for r in row_records) / len(row_records) for m in metric_names}
        summary_df = pd.DataFrame([summary])
        summary_path = out_dir / "metrics_summary_reranker.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary → {summary_path}")

        print("\n=== Averaged Metrics (RERANKER) ===")
        for m, v in summary.items():
            print(f"  {m:<15}: {v:.4f}")

    embedder.close()


if __name__ == "__main__":
    main()
