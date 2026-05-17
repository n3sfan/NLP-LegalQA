import os
import subprocess
import sys
import threading

# Notebook-facing paths for running QA across all row_results configurations.
KAGGLE_INPUT_ROOT = "/kaggle/input/datasets/nesfan/finetune-pk"
DATASET_DIR = (
    f"{KAGGLE_INPUT_ROOT}/eval_results/"
    "eval_results_pipeline_all_ablations_gemma4_rerank_top_30"
)
PAYLOAD_DIR = f"{KAGGLE_INPUT_ROOT}/offline_payloads"
OUTPUT_ROOT = "../eval_results_qa_pipeline_all_ablations_gemma4_rerank_top_30/"
MODEL_NAME = "gemma-4-E2B-it-GGUF"
BASE_PORT = "8002"
PROMPT_TEMPLATE = "prompt_qa_fewshot.md"
WORKDIR = "llm"
LOG_FILE_PATH = "eval_qa_log.txt"


def _dataset_name_from_payload(payload_name: str) -> str | None:
    if not payload_name.startswith("row_results_payload_") or not payload_name.endswith(".jsonl"):
        return None
    suffix = payload_name[len("row_results_payload_") : -len(".jsonl")]
    return f"row_results_{suffix}.csv"


def _build_jobs() -> list[dict[str, str]]:
    payload_names = sorted(
        name for name in os.listdir(PAYLOAD_DIR) if name.startswith("row_results_payload_")
    )
    jobs: list[dict[str, str]] = []

    for payload_name in payload_names:
        dataset_name = _dataset_name_from_payload(payload_name)
        if not dataset_name:
            continue

        dataset_path = os.path.join(DATASET_DIR, dataset_name)
        if not os.path.exists(dataset_path):
            print(f"Skipping payload without matching dataset: {payload_name}")
            continue

        config_name = os.path.splitext(dataset_name)[0]
        output_dir = os.path.join(OUTPUT_ROOT, config_name)
        jobs.append(
            {
                "payload_name": payload_name,
                "dataset_path": dataset_path,
                "output_dir": output_dir,
            }
        )

    return jobs


JOBS = _build_jobs()


def run_evaluation(jobs: list[dict[str, str]], log_file: str) -> None:
    print(f"Starting evaluation thread for {len(jobs)} configs. Logging to: {log_file}")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    with open(log_file, "w", encoding="utf-8") as f:
        for idx, job in enumerate(jobs, 1):
            os.makedirs(job["output_dir"], exist_ok=True)
            cmd = [
                "python",
                "eval_qa_offline.py",
                "--dataset",
                job["dataset_path"],
                "--payload-dir",
                PAYLOAD_DIR,
                "--prompt-template",
                PROMPT_TEMPLATE,
                "--start-index",
                "0",
                "--output",
                job["output_dir"],
                "--models",
                MODEL_NAME,
                "--base-port",
                BASE_PORT,
            ]

            header = f"\n=== [{idx}/{len(jobs)}] Running {os.path.basename(job['dataset_path'])} ===\n"
            f.write(header)
            f.flush()
            print(header, end="")

            process = subprocess.Popen(
                cmd,
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
            footer = (
                f"\nFinished {os.path.basename(job['dataset_path'])} "
                f"with exit code: {process.returncode}\n"
            )
            f.write(footer)
            f.flush()
            print(footer, end="")

            if process.returncode != 0:
                break


eval_thread = threading.Thread(target=run_evaluation, args=(JOBS, LOG_FILE_PATH))
eval_thread.start()

print(f"Evaluation is now running in the background for {len(JOBS)} configs.")
for job in JOBS:
    print(f"- {job['dataset_path']}")
