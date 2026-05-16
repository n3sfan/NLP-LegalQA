"""
Pipeline Evaluation Script with Ablation Testing

Evaluates the current unified retrieval pipeline (retrieve_and_build_context)
on the legal QA dataset.  Supports ablation testing by toggling pipeline
features on/off to measure the impact of each component.

Ablation configurations:
  - full_pipeline : All features ON  (baseline)
  - no_hybrid     : Disable hybrid search (vector only, no BM25)
  - no_heuristic  : Disable heuristic re-ranking (no abolished penalty / recency)
  - agg_borda     : Use Borda aggregation instead of RRF
  - agg_max       : Use Max aggregation instead of RRF

Note: Decompose is ALWAYS ON — the Decomposer prompt normalises colloquial
terms to formal legal terminology (e.g. "vượt đèn đỏ" → "không chấp hành
hiệu lệnh của đèn tín hiệu giao thông").  Disabling it would conflate two
effects (decomposition + term normalisation), making the ablation unfair.

Usage:
    # Run baseline only
    uv run scripts/eval_pipeline.py \\
        --input qa_dataset/QA_Part2.csv \\
        --output eval_results_v4

    # Run all ablation configs
    uv run scripts/eval_pipeline.py \\
        --input qa_dataset/QA_Part2.csv \\
        --output eval_results_v4 \\
        --ablation

    # Run specific configs
    uv run scripts/eval_pipeline.py \\
        --input qa_dataset/QA_Part2.csv \\
        --output eval_results_v4 \\
        --configs full_pipeline no_hybrid

    # Compare with v2 results
    uv run scripts/eval_pipeline.py \
        --input qa_dataset/QA_Part2.csv \
        --output eval_results_v4 \
        --ablation \
        --compare eval_results_v2/metrics_summary_decomposition.csv
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from legal_scraper.embedder import Neo4jEmbedder
from legal_scraper.reranker import VietnameseReranker


# ─────────────────────────────────────────────────────────────────────────────
# Ablation Configurations
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AblationConfig:
    """A single ablation configuration for the retrieval pipeline."""
    name: str
    hybrid: bool = True
    heuristic: bool = True
    expand: bool = False
    aggregate: str = "rrf"

    def describe(self) -> str:
        """Return a human-readable description of this config."""
        flags = []
        flags.append(f"hybrid={'ON' if self.hybrid else 'OFF'}")
        flags.append(f"heuristic={'ON' if self.heuristic else 'OFF'}")
        flags.append(f"expand={'ON' if self.expand else 'OFF'}")
        flags.append(f"aggregate={self.aggregate}")
        return f"{self.name} ({', '.join(flags)})"


ABLATION_CONFIGS = {
    "full_pipeline": AblationConfig("full_pipeline"),
    "no_hybrid": AblationConfig("no_hybrid", hybrid=False),
    "no_heuristic": AblationConfig("no_heuristic", heuristic=False),
    "agg_borda": AblationConfig("agg_borda", aggregate="borda"),
    "agg_max": AblationConfig("agg_max", aggregate="max"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

METRIC_NAMES = [
    "recall@1", "recall@3", "recall@5", "recall@7", "recall@10",
    "precision@1", "precision@3", "mrr",
]


def is_relevant(retrieved_uid: str, reference: str) -> bool:
    """Return True if ``retrieved_uid`` shares a prefix with ``reference``."""
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


def compute_row_metrics(retrieved_uids: list[str], references: list[str]) -> dict:
    """Compute retrieval metrics for a single question."""
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
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_config(
    config: AblationConfig,
    df: pd.DataFrame,
    embedder: Neo4jEmbedder,
    reranker: VietnameseReranker,
    out_dir: Path,
    sleep_seconds: float = 3.0,
) -> dict | None:
    """Run evaluation for a single ablation config. Returns summary dict or None."""

    from legal_scraper.retrieval import retrieve_and_build_context

    row_path = out_dir / f"row_results_{config.name}.csv"
    summary_path = out_dir / f"metrics_summary_{config.name}.csv"

    print(f"\n{'=' * 70}")
    print(f"  CONFIG: {config.describe()}")
    print(f"{'=' * 70}")

    # --- Resume capability ---
    processed_ids: set = set()
    row_records: list[dict] = []

    if row_path.exists():
        try:
            existing_df = pd.read_csv(row_path)
            processed_ids = set(existing_df["id"].tolist())
            row_records = existing_df.to_dict("records")
            print(f"  Resuming from {len(processed_ids)} already processed questions.")
        except Exception as e:
            print(f"  Could not read existing results: {e}")

    skipped = 0
    fallback_count = 0
    t_config_start = time.time()

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

        print(f"\n  [{row_id}/{len(df)}] Q: {question[:80]}{'...' if len(question) > 80 else ''}")

        # --- Run retrieval pipeline ---
        try:
            rr = retrieve_and_build_context(
                embedder=embedder,
                reranker=reranker,
                query=question,
                decompose=True,  # Always ON
                hybrid=config.hybrid,
                aggregate=config.aggregate,
                fetch_k=30,
                rerank_top=30,
                top_k=30,  # Need at least 10 for recall@10
                labels=["Article", "Clause", "Point"],
                expand=config.expand,
                heuristic=config.heuristic,
            )
        except Exception as e:
            # Check if it's an LLM backend error (critical — stop immediately)
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["connection", "timeout", "api", "500", "503"]):
                print(f"\n  [CRITICAL ERROR] LLM/API backend failed at row {idx} (ID: {row_id}).")
                print(f"  Error: {e}")
                print("  Stopping evaluation. Re-run to resume from this row.")
                sys.exit(1)
            # Non-critical error — skip this row
            print(f"  [ERROR] Row {row_id}: {e} — skipped")
            skipped += 1
            continue

        # Extract UIDs from final results
        retrieved_uids = [r.uid for r in rr.final_results]

        if not retrieved_uids:
            fallback_count += 1

        # --- Compute metrics ---
        row_metrics = compute_row_metrics(retrieved_uids, references)
        hit = "✓" if row_metrics["recall@5"] > 0 else "✗"
        print(f"  {hit} R@1={row_metrics['recall@1']:.2f}  R@5={row_metrics['recall@5']:.2f}  MRR={row_metrics['mrr']:.2f}")

        # Log sub-queries
        sub_query_text = " | ".join(rr.sub_queries[:-1]) if rr.sub_queries else ""

        # Log timings
        timing_parts = [f"{k}={v:.1f}s" for k, v in rr.timings.items()]
        if timing_parts:
            print(f"    Timings: {', '.join(timing_parts)}")

        new_record = {
            "id": row_id,
            "question": question,
            "config": config.name,
            "num_subqueries": len(rr.sub_queries) if rr.sub_queries else 0,
            "decomposed_query": sub_query_text,
            "retrieved_uids": ";".join(retrieved_uids),
            "references": ";".join(references),
            **row_metrics,
        }
        row_records.append(new_record)
        processed_ids.add(row_id)

        # Save incrementally (append mode)
        # On resume, file already has header + old rows.
        # On fresh start, first write creates file with header.
        write_header = not row_path.exists()
        pd.DataFrame([new_record]).to_csv(
            row_path, mode='a', header=write_header, index=False,
        )

        if len(processed_ids) % 10 == 0:
            print(f"\n  --- Progress: {len(processed_ids)}/{len(df)} questions processed ---")

        # Give the LLM backend time to clear VRAM
        time.sleep(sleep_seconds)

    t_config_total = time.time() - t_config_start
    print(f"\n  Config '{config.name}' done — {len(row_records)} evaluated, "
          f"{skipped} skipped, {fallback_count} empty results. "
          f"Total time: {t_config_total:.0f}s")

    # --- Save final summary ---
    if not row_records:
        return None

    # Re-save full sorted CSV (clean up incremental appends)
    row_df = pd.DataFrame(row_records)
    row_df.to_csv(row_path, index=False)

    summary = {m: sum(r[m] for r in row_records) / len(row_records) for m in METRIC_NAMES}
    summary["config"] = config.name
    summary["num_evaluated"] = len(row_records)

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {row_path}")
    print(f"  Saved: {summary_path}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Comparison & Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(summaries: list[dict], compare_path: str | None = None):
    """Print a formatted comparison table of all configs."""

    print(f"\n{'=' * 80}")
    print("  ABLATION COMPARISON")
    print(f"{'=' * 80}")

    # Header
    header_metrics = ["recall@1", "recall@3", "recall@5", "recall@10", "precision@1", "mrr"]
    header = f"  {'Config':<20}"
    for m in header_metrics:
        header += f" {m:>10}"
    print(header)
    print(f"  {'─' * (20 + 11 * len(header_metrics))}")

    # Load v2 baseline if provided
    v2_summary = None
    if compare_path and Path(compare_path).exists():
        try:
            v2_df = pd.read_csv(compare_path)
            v2_summary = v2_df.iloc[0].to_dict()
        except Exception:
            pass

    if v2_summary:
        row_str = f"  {'[v2 baseline]':<20}"
        for m in header_metrics:
            val = v2_summary.get(m, 0.0)
            row_str += f" {val:>10.4f}"
        print(row_str)
        print(f"  {'─' * (20 + 11 * len(header_metrics))}")

    # Print each config
    baseline = summaries[0] if summaries else None
    for summary in summaries:
        row_str = f"  {summary['config']:<20}"
        for m in header_metrics:
            val = summary.get(m, 0.0)
            row_str += f" {val:>10.4f}"
        print(row_str)

        # Print delta vs baseline (if not the baseline itself)
        if baseline and summary["config"] != baseline["config"]:
            delta_str = f"  {'  Δ vs baseline':<20}"
            for m in header_metrics:
                delta = summary.get(m, 0.0) - baseline.get(m, 0.0)
                sign = "+" if delta >= 0 else ""
                delta_str += f" {sign}{delta:>9.4f}"
            print(delta_str)

    # Print delta vs v2 for baseline
    if v2_summary and baseline:
        print(f"\n  {'─' * (20 + 11 * len(header_metrics))}")
        delta_str = f"  {'  Δ v4 vs v2':<20}"
        for m in header_metrics:
            v2_val = v2_summary.get(m, 0.0)
            v4_val = baseline.get(m, 0.0)
            delta = v4_val - v2_val
            sign = "+" if delta >= 0 else ""
            delta_str += f" {sign}{delta:>9.4f}"
        print(delta_str)


def save_ablation_comparison(summaries: list[dict], out_dir: Path):
    """Save ablation comparison to CSV."""
    if not summaries:
        return

    comparison_path = out_dir / "ablation_comparison.csv"
    cols = ["config", "num_evaluated"] + METRIC_NAMES
    comp_df = pd.DataFrame(summaries)[cols]
    comp_df.to_csv(comparison_path, index=False)
    print(f"\n  Saved ablation comparison → {comparison_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the unified retrieval pipeline with ablation testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to QA dataset CSV")
    parser.add_argument("--output", default="eval_results_v4", help="Output directory (default: eval_results_v4)")

    # Neo4j connection (defaults from env)
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "neo4j+ssc://localhost:7687"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", ""))
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))

    # Config selection
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "--ablation",
        action="store_true",
        help="Run ALL ablation configs sequentially",
    )
    config_group.add_argument(
        "--configs",
        nargs="+",
        choices=list(ABLATION_CONFIGS.keys()),
        help="Run specific config(s) by name",
    )

    # Optional comparison
    parser.add_argument(
        "--compare",
        default=None,
        help="Path to v2 metrics_summary CSV for comparison",
    )

    # Tuning
    parser.add_argument(
        "--sleep",
        type=float,
        default=3.0,
        help="Seconds to sleep between LLM requests (default: 3.0)",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Determine which configs to run
    if args.ablation:
        config_names = list(ABLATION_CONFIGS.keys())
    elif args.configs:
        config_names = args.configs
    else:
        config_names = ["full_pipeline"]

    configs = [ABLATION_CONFIGS[name] for name in config_names]

    print(f"Pipeline Evaluation — {len(configs)} config(s) to evaluate:")
    for cfg in configs:
        print(f"  • {cfg.describe()}")

    # Load dataset
    df = load_dataset(args.input)
    print(f"\nLoaded {len(df)} questions from {args.input}")

    # Load shared components ONCE
    print("\nInitializing shared components...")
    t_init = time.time()

    embedder = Neo4jEmbedder(args.uri, args.user, args.password, args.database)
    reranker = VietnameseReranker()

    print(f"Components ready ({time.time() - t_init:.1f}s)")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run each config
    summaries: list[dict] = []
    try:
        for cfg in configs:
            summary = evaluate_config(
                config=cfg,
                df=df,
                embedder=embedder,
                reranker=reranker,
                out_dir=out_dir,
                sleep_seconds=args.sleep,
            )
            if summary:
                summaries.append(summary)
    finally:
        embedder.close()

    # Print and save comparison
    if summaries:
        print_summary_table(summaries, args.compare)
        save_ablation_comparison(summaries, out_dir)

    print(f"\nAll done. Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
