"""
Finetuned LLM Evaluation Script

Evaluates the generated answers against ground truth using traditional metrics (BLEU, ROUGE, METEOR)
and LLM-as-a-judge scoring via OpenRouter (google/gemini-2.5-flash).
Generation is done locally using Unsloth and the finetuned model.

Usage:
    # 1. Generate answers
    uv run python scripts/eval_finetuned_llm.py \\
        --step-generate \\
        --ground-truth qa_dataset/QA_Part2.csv \\
        --results eval_results/QA_Part2/eval_results_pipeline_all_ablations_geminiflash_rerank_top_30/row_results_full_pipeline.csv \\
        --output eval_results/QA_Part2/eval_finetuned_results.csv \\
        --model-dir models/gemma-4-E4B-legal-qa-lora
        
    # 2. Run judge
    uv run python scripts/eval_finetuned_llm.py \\
        --step-judge \\
        --ground-truth qa_dataset/QA_Part2.csv \\
        --results eval_results/QA_Part2/eval_results_pipeline_all_ablations_geminiflash_rerank_top_30/row_results_full_pipeline.csv \\
        --output eval_results/QA_Part2/eval_finetuned_results.csv
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
from legal_scraper.prompts import _QA_SYSTEM_PROMPT, _QA_USER_PROMPT, _JUDGE_SYSTEM_PROMPT, _JUDGE_USER_PROMPT

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
    parser = argparse.ArgumentParser(description="Evaluate Finetuned LLM Answers")
    parser.add_argument("--ground-truth", required=True, help="Path to QA dataset CSV (ground truth)")
    parser.add_argument("--results", required=True, help="Path to retrieval results CSV (containing retrieved_uids)")
    parser.add_argument("--output", default="eval_finetuned_results.csv", help="Output CSV path")
    
    parser.add_argument("--step-generate", action="store_true", help="Run generation step with Unsloth")
    parser.add_argument("--step-judge", action="store_true", help="Run LLM-as-a-judge scoring step")
    parser.add_argument("--model-dir", type=str, default="models/gemma-4-E4B-legal-qa-lora", help="Path to the saved LoRA adapters.")
    
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


def run_generate(args, merged_df):
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template
    
    gen_out_path = Path(args.output).with_suffix(".generated.csv")
    gen_out_path.parent.mkdir(parents=True, exist_ok=True)
    processed_ids = set()
    records = []
    
    if gen_out_path.exists():
        try:
            existing_df = pd.read_csv(gen_out_path)
            processed_ids = set(existing_df["id"].tolist())
            records = existing_df.to_dict("records")
            print(f"Resuming generation from {len(processed_ids)} already processed questions.")
        except Exception as e:
            print(f"Could not read existing generation output: {e}")

    print(f"Loading model and LoRA from {args.model_dir}...")
    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=args.model_dir,
            max_seq_length=65000,
            load_in_4bit=True,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("Enabling native 2x faster inference...")
    FastModel.for_inference(model)

    print("Applying Gemma 4 Chat Template...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-4",
    )

    embedder = Neo4jEmbedder(args.uri, args.user, args.password, args.database)
    current_date = datetime.now().strftime("%Y-%m-%d")

    for idx, row in merged_df.iterrows():
        row_id = row["id"]
        if row_id in processed_ids:
            continue
            
        question = row["question"]
        ground_truth = str(row.get("answer", ""))
        retrieved_uids_str = str(row.get("retrieved_uids", ""))
        
        if not question or pd.isna(question):
            print(f"  Row {row_id}: skipped (no question)")
            continue
            
        uids = [u.strip() for u in retrieved_uids_str.split(";") if u.strip()]
        uids = uids[:args.top_k]
        
        print(f"\n[Generate] {row_id}/{len(merged_df)} Q: {question[:80]}...")
        
        # 1. Build context
        if uids:
            try:
                context = build_context_str_for_uids(embedder, uids, expand=args.expand)
            except Exception as e:
                print(f"  [ERROR] Context fetching failed: {e}")
                context = ""
        else:
            context = ""
            
        # Format the user content
        user_content = _QA_SYSTEM_PROMPT + "\n\n" + _QA_USER_PROMPT.format(
            current_date=current_date,
            context=context,
            query=question,
            rewritten_section=""
        )
        
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": user_content
            }]
        }]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to("cuda")

        t0 = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            use_cache=True,
            temperature=1.0, 
            top_p=0.95, 
            top_k=64
        )
        gen_time = time.time() - t0

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Sanitize surrogates
        generated_answer = generated_answer.encode("utf-8", errors="replace").decode("utf-8")

        record = {
            "id": row_id,
            "generated_answer": generated_answer,
            "gen_time": round(gen_time, 2)
        }
        
        records.append(record)
        processed_ids.add(row_id)
        
        # Save incrementally
        write_header = not gen_out_path.exists()
        pd.DataFrame([record]).to_csv(gen_out_path, mode='a', header=write_header, index=False)
        
    embedder.close()
    print(f"\nGeneration complete. {len(processed_ids)} rows processed. Results saved to {gen_out_path}")


def run_judge(args, merged_df):
    gen_out_path = Path(args.output).with_suffix(".generated.csv")
    if not gen_out_path.exists():
        print(f"Error: Generated results not found at {gen_out_path}. Run --step-generate first.")
        sys.exit(1)
        
    gen_df = pd.read_csv(gen_out_path)
    
    # Merge merged_df with gen_df
    eval_df = pd.merge(merged_df, gen_df, on="id", how="inner")
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed_ids = set()
    records = []
    
    if out_path.exists():
        try:
            existing_df = pd.read_csv(out_path)
            processed_ids = set(existing_df["id"].tolist())
            records = existing_df.to_dict("records")
            print(f"Resuming judge from {len(processed_ids)} already processed questions.")
        except Exception as e:
            print(f"Could not read existing judge output: {e}")

    # Initialize components
    print("\nInitializing components for judge...")
    embedder = Neo4jEmbedder(args.uri, args.user, args.password, args.database)
    
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

    print("Starting judge evaluation...")
    skipped = 0

    for idx, row in eval_df.iterrows():
        row_id = row["id"]
        if row_id in processed_ids:
            continue
            
        question = row["question"]
        ground_truth = str(row.get("answer", ""))
        generated_answer = str(row.get("generated_answer", ""))
        gen_time = row.get("gen_time", 0.0)
        retrieved_uids_str = str(row.get("retrieved_uids", ""))
        
        if not question or pd.isna(question):
            skipped += 1
            continue
            
        if not ground_truth or pd.isna(ground_truth) or ground_truth == "nan":
            print(f"  Row {row_id}: skipped (no ground truth answer)")
            skipped += 1
            continue
            
        uids = [u.strip() for u in retrieved_uids_str.split(";") if u.strip()]
        uids = uids[:args.top_k]
        
        print(f"\n[Judge] {row_id}/{len(eval_df)} Q: {question[:80]}...")
        
        # 1. Build context
        if uids:
            try:
                context = build_context_str_for_uids(embedder, uids, expand=args.expand)
            except Exception as e:
                print(f"  [ERROR] Context fetching failed: {e}")
                context = ""
        else:
            context = ""
            
        # 2. Traditional Metrics
        metrics = calculate_metrics(ground_truth, generated_answer)
        
        # 3. LLM-as-a-judge Scoring
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
            "gen_time": gen_time,
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


def main():
    args = parse_args()
    
    if not args.step_generate and not args.step_judge:
        print("Please specify either --step-generate or --step-judge (or both).")
        sys.exit(1)

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
    
    if args.step_generate:
        print("\n=== Running Generation Step ===")
        run_generate(args, merged_df)
        
    if args.step_judge:
        print("\n=== Running Judge Step ===")
        run_judge(args, merged_df)

if __name__ == "__main__":
    main()
