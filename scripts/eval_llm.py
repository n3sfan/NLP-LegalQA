"""
LLM Answer Evaluation Script

Evaluates the generated answers against ground truth using traditional metrics (BLEU, ROUGE, METEOR)
and LLM-as-a-judge scoring via OpenRouter (google/gemini-2.5-flash).

Usage:
    uv run python scripts/eval_llm.py \\
        --ground-truth qa_dataset/QA_Part2.csv \\
        --results eval_results/QA_Part2/eval_results_pipeline_all_ablations_geminiflash_rerank_top_30/row_results_full_pipeline.csv \\
        --output eval_results/QA_Part2/eval_llm_results.csv
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from legal_scraper.embedder import Neo4jEmbedder
from legal_scraper.retrieval import build_context_str_for_uids
from legal_scraper.generator import AnswerGenerator
from legal_scraper.query_rewriter import create_chat_llm
from legal_scraper.prompts import _JUDGE_SYSTEM_PROMPT, _JUDGE_USER_PROMPT

load_dotenv()

# Attempt to download required NLTK resources
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    from nltk.translate.meteor_score import meteor_score
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def calculate_metrics(ref: str, hyp: str) -> dict:
    """Calculate BLEU, ROUGE-L, and METEOR scores."""
    if not ref or not hyp:
        return {"bleu": 0.0, "rougeL": 0.0, "meteor": 0.0}

    ref_tokens = word_tokenize(ref.lower())
    hyp_tokens = word_tokenize(hyp.lower())
    
    # BLEU
    smoothie = SmoothingFunction().method1
    bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
    
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(ref, hyp)
    rouge_l = rouge_scores['rougeL'].fmeasure
    
    # METEOR
    try:
        from nltk.translate.meteor_score import meteor_score
        meteor = meteor_score([ref_tokens], hyp_tokens)
    except Exception:
        meteor = 0.0

    return {
        "bleu": round(bleu, 4),
        "rougeL": round(rouge_l, 4),
        "meteor": round(meteor, 4),
    }

def clean_json_response(text: str) -> str:
    """Clean markdown formatting from LLM JSON output."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation Logic
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLM Answers")
    parser.add_argument("--ground-truth", required=True, help="Path to QA dataset CSV (ground truth)")
    parser.add_argument("--results", required=True, help="Path to retrieval results CSV (containing retrieved_uids)")
    parser.add_argument("--output", default="eval_llm_results.csv", help="Output CSV path")
    parser.add_argument("--top-k", type=int, default=15, help="Number of retrieved UIDs to use for context")
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds to sleep between LLM calls")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to evaluate (for testing)")
    parser.add_argument("--expand", action=argparse.BooleanOptionalAction, default=True, help="Expand context with sibling points and children content")
    
    # Neo4j connection
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "neo4j+ssc://localhost:7687"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", ""))
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading ground truth from: {args.ground_truth}")
    gt_df = pd.read_csv(args.ground_truth)
    
    print(f"Loading retrieval results from: {args.results}")
    res_df = pd.read_csv(args.results)
    
    # Ensure ID mapping works
    if "id" not in res_df.columns:
        print("Warning: 'id' column not found in results dataframe. Resorting to row index.")
        res_df["id"] = range(1, len(res_df) + 1)
    if "id" not in gt_df.columns:
        gt_df["id"] = range(1, len(gt_df) + 1)
        
    # Merge datasets based on 'id'
    merged_df = pd.merge(res_df, gt_df[["id", "answer"]], on="id", how="inner", suffixes=("", "_gt"))
    
    if args.limit is not None:
        merged_df = merged_df.head(args.limit)
    
    print(f"Merged dataframe contains {len(merged_df)} questions.")

    out_path = Path(args.output)
    processed_ids = set()
    records = []
    
    if out_path.exists():
        try:
            existing_df = pd.read_csv(out_path)
            processed_ids = set(existing_df["id"].tolist())
            records = existing_df.to_dict("records")
            print(f"Resuming from {len(processed_ids)} already processed questions.")
        except Exception as e:
            print(f"Could not read existing output: {e}")

    # Initialize components
    print("\nInitializing components...")
    embedder = Neo4jEmbedder(args.uri, args.user, args.password, args.database)
    
    # Generator uses OPENROUTER_MODEL if LLM_PROVIDER=openrouter, else local
    generator = AnswerGenerator()
    
    # Judge uses specific model
    judge_llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="google/gemini-2.5-flash",
        temperature=0.0,
        max_tokens=1024,
    )
    judge_chain = (
        ChatPromptTemplate.from_messages([
            ("system", _JUDGE_SYSTEM_PROMPT),
            ("human", _JUDGE_USER_PROMPT),
        ])
        | judge_llm
        | StrOutputParser()
    )

    print("Starting evaluation...")
    skipped = 0

    for idx, row in merged_df.iterrows():
        row_id = row["id"]
        if row_id in processed_ids:
            continue
            
        question = row["question"]
        ground_truth = str(row.get("answer", ""))
        retrieved_uids_str = str(row.get("retrieved_uids", ""))
        
        if not question or pd.isna(question):
            print(f"  Row {row_id}: skipped (no question)")
            skipped += 1
            continue
            
        if not ground_truth or pd.isna(ground_truth) or ground_truth == "nan":
            print(f"  Row {row_id}: skipped (no ground truth answer)")
            skipped += 1
            continue
            
        uids = [u.strip() for u in retrieved_uids_str.split(";") if u.strip()]
        uids = uids[:args.top_k]
        
        print(f"\n[{row_id}/{len(merged_df)}] Q: {question[:80]}...")
        
        # 1. Build context
        if uids:
            try:
                context = build_context_str_for_uids(embedder, uids, expand=args.expand)
            except Exception as e:
                print(f"  [ERROR] Context fetching failed: {e}")
                context = ""
        else:
            context = ""
            
        # 2. Generate Answer
        t0 = time.time()
        # AnswerGenerator has internal retry via tenacity
        generated_answer = generator.generate_rag_answer(
            query=question, 
            context=context, 
            rewritten_query=None # Single turn
        )
        # Sanitize surrogates (matches logic in api.py)
        generated_answer = generated_answer.encode("utf-8", errors="replace").decode("utf-8")
        gen_time = time.time() - t0
        
        # 3. Traditional Metrics
        metrics = calculate_metrics(ground_truth, generated_answer)
        
        # 4. LLM-as-a-judge Scoring
        t1 = time.time()
        judge_result = {}
        judge_output = ""
        retries = 3
        while retries > 0:
            try:
                judge_output = judge_chain.invoke({
                    "query": question,
                    "context": context,
                    "ground_truth": ground_truth,
                    "generated_answer": generated_answer,
                })
                judge_output = clean_json_response(judge_output)
                judge_result = json.loads(judge_output)
                
                # Verify structure
                _ = judge_result["scores"]["legal_accuracy"]
                break
            except Exception as e:
                print(f"  [WARNING] Judge failed: {e}.")
                print(f"  [DEBUG] Raw output: {judge_output}")
                retries -= 1
                time.sleep(args.sleep * 2)
        
        judge_time = time.time() - t1
        
        if not judge_result:
            print("  [ERROR] Judge failed permanently for this row.")
            judge_result = {
                "reasoning": "Failed to parse judge output.",
                "scores": {"legal_accuracy": 0, "correct_citation": 0, "completeness": 0, "hallucination_citation": 0, "structure": 0}
            }
            
        scores = judge_result.get("scores", {})
        overall_score = (
            scores.get("legal_accuracy", 0) +
            scores.get("correct_citation", 0) +
            scores.get("completeness", 0) +
            scores.get("hallucination_citation", 0) +
            scores.get("structure", 0)
        )
        print(f"  BLEU: {metrics['bleu']:.4f} | ROUGE-L: {metrics['rougeL']:.4f} | Judge: {overall_score}/10")

        record = {
            "id": row_id,
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "bleu": metrics["bleu"],
            "rougeL": metrics["rougeL"],
            "meteor": metrics["meteor"],
            "judge_legal_accuracy": scores.get("legal_accuracy", 0),
            "judge_correct_citation": scores.get("correct_citation", 0),
            "judge_completeness": scores.get("completeness", 0),
            "judge_hallucination_citation": scores.get("hallucination_citation", 0),
            "judge_structure": scores.get("structure", 0),
            "judge_overall_score": overall_score,
            "judge_reasoning": judge_result.get("reasoning", ""),
            "gen_time": round(gen_time, 2),
            "judge_time": round(judge_time, 2),
        }
        
        records.append(record)
        processed_ids.add(row_id)
        
        # Save incrementally
        write_header = not out_path.exists()
        pd.DataFrame([record]).to_csv(out_path, mode='a', header=write_header, index=False)
        
        time.sleep(args.sleep)
        
    embedder.close()
    print(f"\nEvaluation complete. {len(processed_ids)} rows processed. Results saved to {out_path}")
    
    if records:
        # Save final sorted CSV
        pd.DataFrame(records).to_csv(out_path, index=False)
        
        # Calculate summary metrics
        df = pd.DataFrame(records)
        summary = {
            "avg_bleu": df["bleu"].mean(),
            "avg_rougeL": df["rougeL"].mean(),
            "avg_meteor": df["meteor"].mean(),
            "avg_judge_legal_accuracy": df["judge_legal_accuracy"].mean(),
            "avg_judge_correct_citation": df["judge_correct_citation"].mean(),
            "avg_judge_completeness": df["judge_completeness"].mean(),
            "avg_judge_hallucination_citation": df["judge_hallucination_citation"].mean(),
            "avg_judge_structure": df["judge_structure"].mean(),
            "avg_judge_overall": df["judge_overall_score"].mean(),
        }
        
        print("\n--- Summary ---")
        for k, v in summary.items():
            print(f"  {k}: {v:.4f}")
            
if __name__ == "__main__":
    main()
