
import os
os.environ["LD_LIBRARY_PATH"] += ":/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib"

import csv
import json
import logging


import sys
import time
from pathlib import Path
from typing import List, Optional
import pandas as pd
from tqdm import tqdm
import torch
from unsloth import FastLanguageModel
import locale

# Kaggle/Colab UTF-8 locale fix
locale.getpreferredencoding = lambda: "UTF-8"



# Add project root to path
sys.path.append(os.getcwd())

from eval_qa_utils import EvalConfig, load_template, get_payload_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("eval_qa_unsloth")

def run_unsloth_inference(cfg: EvalConfig, model_path: str):
    """Loads offline payloads and performs heavy GPU inference using Unsloth."""
    payload_path = get_payload_path(cfg.dataset_path, cfg.payload_dir)
    if not payload_path.exists():
        log.error("Payload file not found: %s. Run eval_qa_online.py first.", payload_path)
        return

    log.info("Loading payloads from %s", payload_path)
    payloads = []
    with open(payload_path, "r", encoding="utf-8") as f:
        for line in f:
            payloads.append(json.loads(line))
    
    if cfg.limit:
        payloads = payloads[:cfg.limit]
    
    if cfg.start_index > 0:
        payloads = payloads[cfg.start_index:]
        
    n_items = len(payloads)
    log.info("Starting Unsloth inference on %d items", n_items)

    # 1. Load Model
    log.info("Loading Unsloth model from %s (max_seq_length=%d)", model_path, cfg.max_seq_length)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = cfg.max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)


    # 2. Load prompt template
    try:
        qa_template = load_template(cfg.prompt_template_name)
    except FileNotFoundError as e:
        log.error(e)
        return

    # Results CSV setup
    # Use the basename of the model path for the filename
    model_name_clean = Path(model_path).name.replace("/", "_")
    results_path = Path(cfg.output_dir)
    if results_path.suffix != ".csv":
        results_path = results_path / f"row_qa_results_unsloth_{model_name_clean}.csv"
    
    results_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id", "question", "model", "generated_answer", "expert_answer", 
        "latency_ms", "law_text_preview"
    ]

    
    # Write header if not exists or if starting from 0
    if not results_path.exists() or cfg.start_index == 0:
        with open(results_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for idx, item in enumerate(tqdm(payloads, desc="Inferencing"), 1):
        qid = item["id"]
        question = item["question"]
        expert_answer = item["expert_answer"]
        law_text = item["law_text"]
        extra_info = item["extra_info"]

        # Build prompt
        qa_prompt = qa_template.format(question=question, law_text=law_text, extra_info=extra_info)
        
        # Ensure the prompt ends correctly for ChatML / Gemma 4 triggers
        if "<|im_start|>assistant" in qa_prompt:
            if not qa_prompt.strip().endswith("<|im_start|>assistant"):
                qa_prompt = qa_prompt.strip() + "\n<|im_start|>assistant\n"
            elif not qa_prompt.endswith("\n"):
                qa_prompt = qa_prompt.strip() + "\n"
        elif "<|turn>model:" in qa_prompt:
            if not qa_prompt.strip().endswith("<|turn>model:"):
                qa_prompt = qa_prompt.strip() + "\n<|turn>model:\n"
            elif not qa_prompt.endswith("\n"):
                qa_prompt = qa_prompt.strip() + "\n"

        
        # Tokenize and Generate
        start = time.perf_counter()
        # try:
        # Prepend BOS token for Gemma/Llama models to avoid gibberish
        full_prompt = qa_prompt
        inputs = tokenizer(full_prompt, return_tensors = "pt").to("cuda:0")
        input_len = inputs.input_ids.shape[1]
        
        # Using Unsloth's recommended inference parameters
        outputs = model.generate(
            **inputs, 
            max_new_tokens = cfg.max_new_tokens, 
            use_cache = True,
            pad_token_id = tokenizer.eos_token_id,
            do_sample = False,
        )

        
        # Debugging Output Structure
        print(f"DEBUG: QID={qid} | outputs type={type(outputs)}")
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        print(f"DEBUG: sequences shape={sequences.shape} | input_len={input_len}")

        # Extreme safety slice
        if sequences.ndim == 2:
            new_tokens = sequences[0, input_len:]
        else:
            new_tokens = sequences[input_len:]
            
        generated_answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"DEBUG: Generated {len(new_tokens)} tokens")
        print('generated_answer: \n', generated_answer)



        # except Exception as e:
        #     log.error("Model failed for QID %s: %s", qid, e)
        #     generated_answer = f"ERROR: {e}"
            
        duration = (time.perf_counter() - start) * 1000

        # Log Results
        result_row = {
            "id": qid,
            "question": question,
            "model": model_path,
            "generated_answer": generated_answer,
            "expert_answer": expert_answer,
            "latency_ms": round(duration, 1),
            "law_text_preview": law_text[:200] + "..." if law_text else ""
        }
        
        with open(results_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writerow(result_row)

    log.info("Unsloth inference complete. Results saved to %s", results_path)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Unsloth LLM inference for Legal QA")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to local Unsloth model directory")
    parser.add_argument("--dataset", type=str, default="eval_results_v2/row_results_decomposition.csv", help="Original dataset path")
    parser.add_argument("--output", type=str, default="eval_results_qa_unsloth/", help="Output directory")
    parser.add_argument("--payload-dir", type=str, default="offline_payloads/", help="Directory where payloads are stored")
    parser.add_argument("--prompt-template", type=str, default="prompt_qa_fewshot.md", help="QA prompt template filename")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--start-index", type=int, default=0, help="Starting index in payload list")
    parser.add_argument("--print-every", type=int, default=5, help="Logging frequency (for tqdm fallback)")
    parser.add_argument("--max-seq-length", type=int, default=65536, help="Maximum context window size")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Maximum new tokens to generate")

    args = parser.parse_args()
    
    cfg = EvalConfig(
        dataset_path=args.dataset,
        output_dir=args.output,
        payload_dir=args.payload_dir,
        prompt_template_name=args.prompt_template,
        limit=args.limit,
        start_index=args.start_index,
        print_every=args.print_every,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens
    )


    
    run_unsloth_inference(cfg, args.model_dir)

if __name__ == "__main__":
    main()
