import asyncio
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm.auto import tqdm


# Add project root to path
sys.path.append(os.getcwd())

from voter import OpenRouterBackend, VLLMBackend
from eval_qa_utils import EvalConfig, load_template, get_payload_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("eval_qa_offline")

# Silence noisy third-party logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

DEFAULT_VLLM_MODEL = "Qwen/Qwen3-4B"
DEFAULT_OPENROUTER_MODEL = "google/gemma-4-26b-a4b-it:free"


def _resolve_model_names(cfg: EvalConfig) -> list[str]:
    if cfg.backend_type == "openrouter" and cfg.models == [DEFAULT_VLLM_MODEL]:
        return [os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)]
    return cfg.models


def _build_backends(cfg: EvalConfig):
    model_names = _resolve_model_names(cfg)
    backends = []

    if cfg.backend_type == "openrouter":
        api_key = cfg.api_key
        if not api_key or api_key == "vllm-secret-key":
            api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY must be set when using --openrouter")

        for model_name in model_names:
            log.info("Initializing OpenRouter model %s", model_name)
            backends.append(OpenRouterBackend(model=model_name, api_key=api_key))
    else:
        for i, model_name in enumerate(model_names):
            port = cfg.base_port + i
            url = f"http://localhost:{port}/v1"
            log.info("Initializing model %s at %s", model_name, url)
            backends.append(VLLMBackend(model=model_name, base_url=url, api_key=cfg.api_key))

    return model_names, backends


async def run_offline_inference(cfg: EvalConfig):
    """Loads offline payloads and performs heavy GPU inference."""
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
    log.info("Starting offline inference on %d items", n_items)

    # Initialize Backends
    try:
        models, backends = _build_backends(cfg)
    except ValueError as e:
        log.error(e)
        return

    # Load prompt template
    try:
        qa_template = load_template(cfg.prompt_template_name)
    except FileNotFoundError as e:
        log.error(e)
        return

    # Results CSV setup
    output_p = Path(cfg.output_dir)
    output_p.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "id", "question", "model", "generated_answer", "expert_answer", 
        "latency_ms", "law_text_preview"
    ]
    results_path = output_p / "row_qa_results_offline.csv"
    
    # Write header if not exists or if starting from 0
    if not results_path.exists() or cfg.start_index == 0:
        with open(results_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for idx, item in enumerate(tqdm(payloads, desc="Offline Inference"), 1):
        qid = item["id"]
        question = item["question"]
        expert_answer = item["expert_answer"]
        law_text = item["law_text"]
        extra_info = item["extra_info"]

        # if (idx-1) % cfg.print_every == 0 or idx == n_items:
        #     log.info("[%d/%d] Generating answer for QID: %s", idx, n_items, qid)

        # Build prompt
        # We use a dict to safely format only the keys present in the template
        prompt_kwargs = {
            "question": question, 
            "law_text": law_text, 
            "extra_info": extra_info,
        }
        # Filter kwargs to only those present in the template to avoid KeyError
        needed_keys = re.findall(r"\{(\w+)\}", qa_template)
        filtered_kwargs = {k: v for k, v in prompt_kwargs.items() if k in needed_keys}
        qa_prompt = qa_template.format(**filtered_kwargs)
        
        async def ask_model(backend, model_name):
            current_prompt = qa_prompt
            if "gemma" in model_name.lower():
                # Specific mappings for Gemma 3 / Fine-tuned templates
                current_prompt = current_prompt.replace("<|im_start|>user", "<|turn>user")
                current_prompt = current_prompt.replace("<|im_start|>assistant", "<|turn>model")
                # Fallback / General tokens
                current_prompt = current_prompt.replace("<|im_start|>", "<|turn>")
                current_prompt = current_prompt.replace("<|im_end|>", "<turn|>")
                # Thinking / Reasoning tokens
                current_prompt = current_prompt.replace("<think>", "<|channel>thought")
                current_prompt = current_prompt.replace("</think>", "<channel|>")
            
            max_retries = 3
            for attempt in range(max_retries):
                start = time.perf_counter()
                try:
                    ans = await backend.ask(current_prompt)
                    duration = (time.perf_counter() - start) * 1000
                    if ans and ans.strip():
                        return model_name, ans, duration
                    else:
                        log.warning("Empty response from %s (attempt %d/%d)", model_name, attempt+1, max_retries)
                except Exception as e:
                    log.error("Model %s failed for QID %s (attempt %d/%d): %s", model_name, qid, attempt+1, max_retries, e)
                    ans = f"ERROR: {e}"
                    duration = (time.perf_counter() - start) * 1000
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(1) # Small backoff
            
            return model_name, ans, duration

        tasks = [ask_model(b, m) for b, m in zip(backends, models)]
        results = await asyncio.gather(*tasks)

        # Log Results
        for model_name, generated_answer, duration in results:
            result_row = {
                "id": qid,
                "question": question,
                "model": model_name,
                "generated_answer": generated_answer,
                "expert_answer": expert_answer,
                "latency_ms": round(duration, 1),
                "law_text_preview": law_text[:200] + "..." if law_text else ""
            }
            
            with open(results_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writerow(result_row)

    log.info("Offline inference complete. Results saved to %s", results_path)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run offline LLM inference for Legal QA")
    parser.add_argument("--dataset", type=str, default="eval_results_v2/row_results_decomposition.csv", help="Original dataset path (used to find payload)")
    parser.add_argument("--output", type=str, default="eval_results_qa_offline/", help="Output directory")
    parser.add_argument("--payload-dir", type=str, default="offline_payloads/", help="Directory where payloads are stored")
    parser.add_argument("--prompt-template", type=str, default="prompt_qa_0shot.md", help="QA prompt template filename")
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen3-4B"], help="List of model names")
    parser.add_argument("--api-key", type=str, default="vllm-secret-key", help="API key")
    parser.add_argument("--base-port", type=int, default=8080, help="Starting port for vllm")
    parser.add_argument("--openrouter", action="store_true", help="Use OpenRouter instead of local vLLM ports")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--start-index", type=int, default=0, help="Starting index in payload list")
    parser.add_argument("--print-every", type=int, default=5, help="Logging frequency")
    # parser.add_argument("--top-k", type=int, default=5, help="Top-K context size (fallback if not in payload)")

    args = parser.parse_args()
    
    cfg = EvalConfig(
        dataset_path=args.dataset,
        output_dir=args.output,
        payload_dir=args.payload_dir,
        prompt_template_name=args.prompt_template,
        backend_type="openrouter" if args.openrouter else "vllm",
        models=args.models,
        api_key=args.api_key,
        base_port=args.base_port,
        limit=args.limit,
        start_index=args.start_index,
        print_every=args.print_every,
        top_k=args.top_k
    )
    
    asyncio.run(run_offline_inference(cfg))

if __name__ == "__main__":
    main()
