import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm.asyncio import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from eval_voter import fetch_law_texts
from legal_scraper.embedder import Neo4jEmbedder
from eval_qa_utils import EvalConfig, load_template, is_relevant, get_payload_path, get_val

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("eval_qa_online")

async def generate_payload(cfg: EvalConfig):
    """Fetches law context and saves it to an offline payload file."""
    dataset_p = Path(cfg.dataset_path)
    if not dataset_p.exists():
        log.error("Dataset not found: %s", cfg.dataset_path)
        return

    log.info("Loading dataset from %s", cfg.dataset_path)
    df = pd.read_csv(dataset_p)
    
    if cfg.limit:
        df = df.head(cfg.limit)
    
    if cfg.start_index > 0:
        df = df.iloc[cfg.start_index:]
    
    n_questions = len(df)
    log.info("Preparing payloads for %d questions (from index %d)", n_questions, cfg.start_index)

    # Initialize Neo4j
    embedder = Neo4jEmbedder(
        uri=cfg.neo4j_uri, 
        user=cfg.neo4j_user, 
        password=cfg.neo4j_password, 
        database=cfg.neo4j_database
    )
    
    payload_dir = Path(cfg.payload_dir)
    payload_dir.mkdir(parents=True, exist_ok=True)
    payload_path = get_payload_path(cfg.dataset_path, cfg.payload_dir)
    
    # Mode 'a' to support resuming
    with open(payload_path, "a", encoding="utf-8") as f:
        try:
            for idx, row in enumerate(df.itertuples(), 1):
                qid = get_val(row, "id", str(idx))
                question = get_val(row, "question")
                expert_answer = get_val(row, "answer", get_val(row, "references"))
                
                # Parse UIDs (top k retrieved or references)
                if hasattr(row, "retrieved_uids"):
                    raw_retrieved = getattr(row, "retrieved_uids", "")
                    if pd.isna(raw_retrieved): raw_retrieved = ""
                    uids = [r.strip() for r in str(raw_retrieved).split(";") if r.strip()][:cfg.top_k]
                else:
                    raw_refs = get_val(row, "reference", get_val(row, "references"))
                    uids = [r.strip() for r in raw_refs.replace(";", ",").split(",") if r.strip()]

                # Parse ground-truth references
                ref_list = [r.strip() for r in get_val(row, "references").replace(";", ",").split(",") if r.strip()]
                
                # Check Recall@k == 1.0 logic
                found_refs = {ref for u in uids for ref in ref_list if is_relevant(u, ref)}
                if ref_list and len(found_refs) < len(ref_list):
                    if (idx-1) % cfg.print_every == 0 or idx == n_questions:
                        log.info("[%d/%d] Skipping QID: %s (Recall@%d < 1.0)", idx, n_questions, qid, cfg.top_k)
                    continue

                if (idx-1) % cfg.print_every == 0 or idx == n_questions:
                    log.info("[%d/%d] Fetching context for QID: %s", idx, n_questions, qid)

                # Fetch Context from Neo4j
                _, _, law_text, extra_info = fetch_law_texts(embedder, uids, batch_size=cfg.batch_size)

                payload_item = {
                    "id": qid,
                    "question": question,
                    "expert_answer": expert_answer,
                    "uids": uids,
                    "law_text": law_text or "",
                    "extra_info": extra_info or "",
                    "top_k": cfg.top_k
                }
                f.write(json.dumps(payload_item, ensure_ascii=False) + "\n")
                
        finally:
            embedder.close()

    log.info("Payload generation complete. Saved to %s", payload_path)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate offline payloads for Legal QA")
    parser.add_argument("--dataset", type=str, default="eval_results_v2/row_results_decomposition.csv", help="QA dataset path")
    parser.add_argument("--payload-dir", type=str, default="offline_payloads/", help="Directory to save payloads")
    parser.add_argument("--uri", type=str, default="neo4j+ssc://nguyenhoangquan.com:7687", help="Neo4j URI")
    parser.add_argument("--user", type=str, default="neo4j", help="Neo4j user")
    parser.add_argument("--password", type=str, default="Neoneo4j", help="Neo4j password")
    parser.add_argument("--database", type=str, default="neo4j", help="Neo4j database name")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--start-index", type=int, default=0, help="Starting row index")
    parser.add_argument("--print-every", type=int, default=5, help="Logging frequency")
    parser.add_argument("--batch-size", type=int, default=10, help="Neo4j batch size")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K retrieved UIDs to include")

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
