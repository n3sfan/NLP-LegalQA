import asyncio
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from neo4j import GraphDatabase
from tqdm.asyncio import tqdm

# Add project root to path to ensure imports work
sys.path.append(os.getcwd())

from voter import VLLMBackend
from eval_voter import fetch_law_texts

# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("eval_qa")

# ─────────────────────────────────────────────────────────────────────────────
# Prompt Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_template(filename: str) -> str:
    # Try current dir first, then src/llm/
    p = Path(filename)
    if not p.exists():
        p = Path("src/llm") / filename
    if not p.exists():
        # Fallback to absolute path search if needed
        p = Path(__file__).parent / filename
    if not p.exists():
        raise FileNotFoundError(f"Template not found: {filename}")
    return p.read_text(encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# Core Logic
# ─────────────────────────────────────────────────────────────────────────────

async def evaluate_qa(
    dataset_path: str,
    output_dir: str,
    prompt_template_name: str,
    base_port: int,
    models: List[str],
    api_key: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
    batch_size: int = 10,
    limit: Optional[int] = None,
    print_every: int = 5,
):
    """Main evaluation loop for Legal QA using ground truth references."""
    output_p = Path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)

    log.info("Loading QA dataset from %s", dataset_path)
    qa_p = Path(dataset_path)
    if not qa_p.exists():
        log.error("QA dataset not found: %s", dataset_path)
        return
    df = pd.read_csv(qa_p)
    
    if limit:
        df = df.head(limit)
    
    n_questions = len(df)
    log.info("Starting QA generation on %d questions", n_questions)

    # Initialize Neo4j driver
    driver = GraphDatabase.driver(
        neo4j_uri, 
        auth=(neo4j_user, neo4j_password),
        database=neo4j_database
    )

    # Initialize Backends (Increment port for each model like eval_voter.py)
    backends = []
    for i, m in enumerate(models):
        port = base_port + i
        url = f"http://localhost:{port}/v1"
        log.info("Initializing model %s at %s", m, url)
        backends.append(VLLMBackend(model=m, base_url=url, api_key=api_key))

    # Load templates
    try:
        qa_template = load_template(prompt_template_name)
    except FileNotFoundError as e:
        log.error(e)
        return

    # Results CSV setup
    fieldnames = [
        "id", "question", "model", "generated_answer", "expert_answer", 
        "latency_ms", "law_text_preview"
    ]
    results_path = output_p / "row_qa_results.csv"
    
    # Write header
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for idx, row in enumerate(df.itertuples(), 1):
        qid = str(row.id)
        question = str(row.question)
        expert_answer = str(row.answer)
        
        # Parse reference UIDs (ground truth)
        raw_refs = str(getattr(row, "reference", "")).strip()
        # Split by comma or semicolon
        uids = [r.strip() for r in raw_refs.replace(";", ",").split(",") if r.strip()]

        if (idx-1) % print_every == 0 or idx == n_questions:
            log.info("[%d/%d] Processing QID: %s", idx, n_questions, qid)

        # 1. Fetch Law Text
        uid_to_text, _ = fetch_law_texts(driver, uids, batch_size=batch_size)
        # Join all referenced law texts
        law_text = "\n\n".join(uid_to_text.values())

        if not law_text:
            log.warning("No law text found for QID %s (references: %s)", qid, uids)
            law_text = ""

        # 2. QA (Answer Generation) - Run all models in parallel
        qa_prompt = qa_template.format(question=question, law_text=law_text)
        
        async def ask_model(backend, model_name):
            start = time.perf_counter()
            try:
                print('prompt', qa_prompt)
                ans = await backend.ask(qa_prompt)
                print('ans', ans)
            except Exception as e:
                log.error("Model %s failed for QID %s: %s", model_name, qid, e)
                ans = f"ERROR: {e}"
            duration = (time.perf_counter() - start) * 1000
            return model_name, ans, duration

        tasks = [ask_model(b, m) for b, m in zip(backends, models)]
        results = await asyncio.gather(*tasks)

        # 3. Log Results
        for model_name, generated_answer, duration in results:
            result_row = {
                "id": qid,
                "question": question,
                "model": model_name,
                "generated_answer": generated_answer,
                "expert_answer": expert_answer,
                "latency_ms": round(duration, 1),
                "law_text_preview": law_text[:500].replace("\n", " ") if law_text else ""
            }
            
            with open(results_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writerow(result_row)

    driver.close()
    log.info("QA Generation complete. Results saved to %s", results_path)

# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Legal QA answers from ground truth references")
    parser.add_argument("--dataset", type=str, default="qa_dataset/QA_NLP.csv", help="QA dataset path")
    parser.add_argument("--output", type=str, default="eval_results/", help="Output directory")
    parser.add_argument("--prompt-template", type=str, default="prompt_qa_0shot.md", help="QA prompt template filename")
    parser.add_argument("--print-every", type=int, default=5, help="Logging frequency")
    parser.add_argument("--backend", type=str, default="vllm", help="Backend type")
    parser.add_argument("--n-voters", type=int, default=1, help="Number of models to use")
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen3-4B"], help="List of model names")
    parser.add_argument("--api-key", type=str, default="vllm-secret-key", help="API key")
    parser.add_argument("--base-port", type=int, default=8080, help="Starting port for vllm voters (voter i -> localhost:{base-port+i})")
    parser.add_argument("--uri", type=str, default="neo4j+ssc://nguyenhoangquan.com:7687", help="Neo4j URI")
    parser.add_argument("--user", type=str, default="neo4j", help="Neo4j user")
    parser.add_argument("--password", type=str, default="Neoneo4j", help="Neo4j password")
    parser.add_argument("--database", type=str, default="neo4j", help="Neo4j database name")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")

    args = parser.parse_args()

    asyncio.run(evaluate_qa(
        dataset_path=args.dataset,
        output_dir=args.output,
        prompt_template_name=args.prompt_template,
        base_port=args.base_port,
        models=args.models,
        api_key=args.api_key,
        neo4j_uri=args.uri,
        neo4j_user=args.user,
        neo4j_password=args.password,
        neo4j_database=args.database,
        limit=args.limit,
        print_every=args.print_every
    ))

if __name__ == "__main__":
    main()
