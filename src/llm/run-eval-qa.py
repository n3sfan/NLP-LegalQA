import os
import subprocess
import sys
import threading

# Notebook-facing paths for judging QA answers across all row_results configs.
KAGGLE_INPUT_ROOT = "/kaggle/input/datasets/nesfan/finetune-pk"
DATASET_DIR = (
    f"{KAGGLE_INPUT_ROOT}/eval_results/"
    "eval_results_pipeline_all_ablations_gemma4_rerank_top_30"
)
PAYLOAD_DIR = f"{KAGGLE_INPUT_ROOT}/offline_payloads"
GT_DATASET = f"{KAGGLE_INPUT_ROOT}/qa_dataset/QA_NLP.csv"
QA_RESULTS_ROOT = "../eval_results_qa_pipeline_all_ablations_gemma4_rerank_top_30/"
OUTPUT_ROOT = "../eval_results_qa_eval_pipeline_all_ablations_gemma4_rerank_top_30/"
MODEL_NAME = "gemma-4-E2B-it-GGUF"
BASE_PORT = "8002"
PROMPT_TEMPLATE = "prompt_eval_qa_fewshot.md"
WORKDIR = "llm"
LOG_FILE_PATH = "eval_llm_answer_qa_log.txt"

cmd = [
    "python",
    "eval_voter.py",
    "--mode",
    "eval_qa",
    "--backend",
    "vllm",
    "--models",
    MODEL_NAME,
    "--base-port",
    BASE_PORT,
    "--input",
    QA_RESULTS_ROOT,
    "--payload-dir",
    PAYLOAD_DIR,
    "--dataset",
    DATASET_DIR,
    "--prompt-template",
    PROMPT_TEMPLATE,
    "--gt-dataset",
    GT_DATASET,
    "--output",
    OUTPUT_ROOT,
]


def run_evaluation(command: list[str], log_file: str) -> None:
    print(f"Starting QA judge evaluation thread. Logging to: {log_file}")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    with open(log_file, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            command,
            cwd=WORKDIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        for line in process.stdout:
            f.write(line)
            f.flush()
            print(line, end="")
            sys.stdout.flush()

        process.wait()
        f.write(f"\nProcess finished with exit code: {process.returncode}\n")
        print(f"\nProcess finished with exit code: {process.returncode}")


eval_thread = threading.Thread(target=run_evaluation, args=(cmd, LOG_FILE_PATH))
eval_thread.start()

print(f"QA judge evaluation is now running in the background for: {DATASET_DIR}")
