"""
RAG Retrieval Evaluation Script

Evaluates Neo4j vector search retrieval on the legal QA dataset.

Usage:
    uv run scripts/eval_rag.py \
        --input qa_dataset/QA_NLP.csv \
        --output eval_results \
        --uri "neo4j+ssc://host:7687" \
        --user neo4j \
        --password "..." \
        --database neo4j

Credentials can also be set via environment variables:
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
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

from langchain_neo4j import Neo4jVector
from legal_scraper.embedder import VietnameseEmbeddings


# ─────────────────────────────────────────────────────────────────────────────
# Relevance
# ─────────────────────────────────────────────────────────────────────────────

def is_relevant(retrieved_uid: str, reference: str) -> bool:
    """Return True if `retrieved_uid` shares a prefix with `reference`."""
    return retrieved_uid.startswith(reference)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

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
    """Compute all metrics for a single row."""
    total_relevant = len(references)

    metrics = {}

    for k in [1, 3, 5, 7, 10]:
        top_k = retrieved_uids[:k]
        # Count unique references covered — a reference is "found" once
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
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval on legal QA dataset")
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
    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"question", "reference"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def parse_reference(ref_str: str) -> list[str]:
    """Split comma-separated references and strip whitespace."""
    if not isinstance(ref_str, str) or not ref_str.strip():
        return []
    return [r.strip() for r in ref_str.split(",") if r.strip()]


def search_top30(
    query: str,
    vector_stores: dict[str, Neo4jVector],
    k_per_label: int = 30,
) -> list[tuple[str, str, float]]:
    """
    Query each label index and merge the top-k_per_label results into a
    single globally-ranked list of (uid, label, score).
    """
    all_results: list[tuple[str, str, float]] = []
    for label, vector in vector_stores.items():
        docs = vector.similarity_search_with_score(query, k=k_per_label)
        for doc, score in docs:
            uid = doc.metadata.get("uid", "")
            all_results.append((uid, label, score))
    # Sort by score descending (higher cosine similarity = better match)
    all_results.sort(key=lambda x: x[2], reverse=True)
    return all_results[:k_per_label]


def main():
    args = parse_args()

    # Load dataset
    df = load_dataset(args.input)
    print(f"Loaded {len(df)} questions from {args.input}")

    # Create shared embedding model once
    print("Loading embedding model ...")
    embedding_model = VietnameseEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Build one Neo4jVector per label, reusing the same embedding model
    labels = ["Article", "Clause", "Point"]
    vector_stores: dict[str, Neo4jVector] = {}
    for label in labels:
        vector_stores[label] = Neo4jVector.from_existing_index(
            embedding=embedding_model,
            url=args.uri,
            username=args.user,
            password=args.password,
            database=args.database,
            index_name=f"{label}_embedding_index",
            text_node_properties=["title", "content"] if label == "Article" else ["content"],
            embedding_node_property="embedding",
        )
        print(f"  Index '{label}_embedding_index' ready.")

    # Prepare output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-row results
    row_records = []
    metric_names = ["recall@1", "recall@3", "recall@5", "recall@7", "recall@10",
                    "precision@1", "precision@3",
                    "mrr"]

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

        # Retrieve top-30 from merged Article/Clause/Point indexes
        results = search_top30(question, vector_stores, k_per_label=30)
        retrieved_uids = [uid for uid, _, _ in results]

        # Compute metrics
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

    # ── Save per-row results ──────────────────────────────────────────────────
    row_df = pd.DataFrame(row_records, columns=["id", "question", "retrieved_uids", "references"] + metric_names)
    row_path = out_dir / "row_results.csv"
    row_df.to_csv(row_path, index=False)
    print(f"Saved per-row results → {row_path}")

    # ── Save summary (averaged metrics) ───────────────────────────────────────
    if row_records:
        summary = {m: sum(r[m] for r in row_records) / len(row_records) for m in metric_names}
        summary_df = pd.DataFrame([summary])
        summary_path = out_dir / "metrics_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary → {summary_path}")

        print("\n=== Averaged Metrics ===")
        for m, v in summary.items():
            print(f"  {m:<15}: {v:.4f}")


if __name__ == "__main__":
    main()
