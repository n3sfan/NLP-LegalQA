import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Add project root to path
sys.path.append(os.getcwd())

from legal_scraper.embedder import Neo4jEmbedder
from legal_scraper.retrieval import build_context_str_for_uids
from eval_qa_utils import EvalConfig, is_relevant, get_payload_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("eval_qa_online")

def _discover_datasets(dataset_path: str) -> list[Path]:
    """Return one CSV path or all row_results CSVs under a directory."""
    dataset_p = Path(dataset_path)
    if not dataset_p.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if dataset_p.is_file():
        return [dataset_p]
    return sorted(dataset_p.glob("row_results*.csv"))


def _row_value(row: pd.Series, *names: str, default: str = "") -> str:
    for name in names:
        if name in row and pd.notna(row[name]):
            return str(row[name]).strip()
    return default


def _parse_uids(raw: Any, separator: str = ";") -> list[str]:
    if raw is None or pd.isna(raw):
        return []
    return [uid.strip() for uid in str(raw).split(separator) if uid.strip()]


def _trim_uids(uids: list[str], top_k: int) -> list[str]:
    if top_k < 0:
        return uids
    return uids[:top_k]


def _find_recall_column(df: pd.DataFrame, top_k: int) -> tuple[int, str | None]:
    recall_candidates = []
    for col in df.columns:
        match = re.fullmatch(r"recall@(\d+)", str(col))
        if not match:
            continue
        recall_at = int(match.group(1))
        if top_k < 0 or recall_at <= top_k:
            recall_candidates.append((recall_at, str(col)))
    return max(recall_candidates, default=(top_k, None))


def _infer_expand(dataset_p: Path, df: pd.DataFrame) -> bool:
    """Mirror an explicit expansion marker when future row_results include one."""
    if "expand" in df.columns:
        values = df["expand"].dropna().astype(str).str.lower()
        return values.isin(["1", "true", "yes", "on"]).any()
    config_name = dataset_p.stem.lower()
    return "expand" in config_name and "no_expand" not in config_name


def _process_dataset(dataset_p: Path, cfg: EvalConfig, embedder: Neo4jEmbedder) -> None:
    log.info("Loading dataset from %s", dataset_p)
    df = pd.read_csv(dataset_p)

    if cfg.limit:
        df = df.head(cfg.limit)

    if cfg.start_index > 0:
        df = df.iloc[cfg.start_index:]

    recall_cutoff, recall_col = _find_recall_column(df, cfg.top_k)
    expand = _infer_expand(dataset_p, df)

    n_questions = len(df)
    log.info(
        "Preparing payloads for %d questions from %s (start_index=%d, expand=%s)",
        n_questions,
        dataset_p.name,
        cfg.start_index,
        expand,
    )

    payload_path = get_payload_path(str(dataset_p), cfg.payload_dir)

    # Mode 'a' to support resuming
    with open(payload_path, "a", encoding="utf-8") as f:
        for idx, (_, row) in enumerate(df.iterrows(), 1):
            qid = _row_value(row, "id", default=str(idx))
            question = _row_value(row, "question")
            expert_answer = _row_value(row, "answer", "expert_answer", "references")

            # Parse UIDs from final retrieval output when available.
            if "retrieved_uids" in row:
                uids = _trim_uids(_parse_uids(row["retrieved_uids"], separator=";"), cfg.top_k)
            else:
                raw_refs = _row_value(row, "reference", "references")
                uids = _trim_uids(
                    [r.strip() for r in raw_refs.replace(";", ",").split(",") if r.strip()],
                    cfg.top_k,
                )

            # Parse ground-truth references for optional recall filtering.
            raw_ref_list = _row_value(row, "references", "reference")
            ref_list = [r.strip() for r in raw_ref_list.replace(";", ",").split(",") if r.strip()]

            # Prefer the nearest available recall@N column where N <= top_k.
            # Fall back to exact top-k matching when per-row recall columns are absent.
            if ref_list and recall_col:
                recall_value = row.get(recall_col)
                recall_check_failed = pd.isna(recall_value) or float(recall_value) < 1.0
            elif ref_list:
                found_refs = {ref for uid in uids for ref in ref_list if is_relevant(uid, ref)}
                recall_check_failed = len(found_refs) < len(ref_list)
            else:
                recall_check_failed = False

            if recall_check_failed:
                if (idx - 1) % cfg.print_every == 0 or idx == n_questions:
                    log.info(
                        "[%d/%d] Skipping QID: %s (Recall@%d < 1.0)",
                        idx,
                        n_questions,
                        qid,
                        recall_cutoff,
                    )
                continue

            if (idx - 1) % cfg.print_every == 0 or idx == n_questions:
                log.info("[%d/%d] Fetching context for QID: %s", idx, n_questions, qid)

            # Build the same context style as retrieve_and_build_context().context_str.
            law_text = build_context_str_for_uids(embedder, uids, expand=expand)
            extra_info = ""

            payload_item = {
                "id": qid,
                "question": question,
                "expert_answer": expert_answer,
                "uids": uids,
                "law_text": law_text or "",
                "extra_info": extra_info or "",
                "top_k": cfg.top_k,
            }
            f.write(json.dumps(payload_item, ensure_ascii=False) + "\n")

    log.info("Payload generation complete. Saved to %s", payload_path)


async def generate_payload(cfg: EvalConfig):
    """Fetch law context and save it to offline payload file(s)."""
    try:
        datasets = _discover_datasets(cfg.dataset_path)
    except FileNotFoundError as exc:
        log.error(exc)
        return

    if not datasets:
        log.error("No row_results*.csv files found under %s", cfg.dataset_path)
        return

    payload_dir = Path(cfg.payload_dir)
    if payload_dir.suffix == ".jsonl" and len(datasets) > 1:
        log.error("--payload-dir must be a directory when --dataset points to multiple CSVs")
        return
    if not payload_dir.suffix:
        payload_dir.mkdir(parents=True, exist_ok=True)

    embedder = Neo4jEmbedder(
        uri=cfg.neo4j_uri,
        user=cfg.neo4j_user,
        password=cfg.neo4j_password,
        database=cfg.neo4j_database,
    )

    try:
        for dataset_p in datasets:
            _process_dataset(dataset_p, cfg, embedder)
    finally:
        embedder.close()

    log.info("All payload generation complete (%d dataset(s)).", len(datasets))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate offline payloads for Legal QA")
    parser.add_argument("--dataset", type=str, default="eval_results_v2/row_results_decomposition.csv", help="QA row_results CSV path or directory")
    parser.add_argument("--payload-dir", type=str, default="offline_payloads/", help="Directory to save payloads, or a .jsonl path for one CSV")
    parser.add_argument("--top-k", type=int, default=30, help="Top-K retrieved UIDs to include")

    parser.add_argument("--uri", type=str, default="neo4j+ssc://nguyenhoangquan.com:7687", help="Neo4j URI")
    parser.add_argument("--user", type=str, default="neo4j", help="Neo4j user")
    parser.add_argument("--password", type=str, default="Neoneo4j", help="Neo4j password")
    parser.add_argument("--database", type=str, default="neo4j", help="Neo4j database name")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--start-index", type=int, default=0, help="Starting row index")
    parser.add_argument("--print-every", type=int, default=5, help="Logging frequency")
    parser.add_argument("--batch-size", type=int, default=10, help="Neo4j batch size")

    args = parser.parse_args()
    
    cfg = EvalConfig(
        dataset_path=args.dataset,
        payload_dir=args.payload_dir,
        neo4j_uri=args.uri,
        neo4j_user=args.user,
        neo4j_password=args.password,
        neo4j_database=args.database,
        limit=args.limit,
        start_index=args.start_index,
        print_every=args.print_every,
        batch_size=args.batch_size,
        top_k=args.top_k
    )
    
    asyncio.run(generate_payload(cfg))

if __name__ == "__main__":
    main()
