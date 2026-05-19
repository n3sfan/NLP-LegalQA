import os
import subprocess
import sys
import threading

MODEL_NAME = "gemma-4-E2B-it"
OUTPUT_ROOT = f"../eval_results_pipeline_all_ablations_geminiflash_rerank_top_30/llm/{MODEL_NAME}"
LOG_FILE_PATH = "qa_log_finetune.txt"

KAGGLE_INPUT_ROOT = "/kaggle/input/datasets/nesfan/finetune-pk"
DATASET_DIR = (
    f"{KAGGLE_INPUT_ROOT}/"
    "eval_results_pipeline_all_ablations_geminiflash_rerank_top_30"
)
PAYLOAD_DIR = f"{KAGGLE_INPUT_ROOT}/offline_payloads"
BASE_PORT = "8002"
PROMPT_TEMPLATE = "prompts/prompt_qa_0shot_quan.md"
CONFIGS = ["full_pipeline"]
WORKDIR = "llm"


def _dataset_name_from_payload(payload_name: str) -> str | None:
    if not payload_name.endswith(".jsonl"):
        return None

    if payload_name.startswith("row_results_payload_"):
        suffix = payload_name[len("row_results_payload_") : -len(".jsonl")]
        return f"row_results_{suffix}.csv"

    if payload_name.startswith("row_results_") and payload_name.endswith("_payload.jsonl"):
        return payload_name[: -len("_payload.jsonl")] + ".csv"

    return None


def _config_name_from_dataset_name(dataset_name: str) -> str:
    stem = os.path.splitext(dataset_name)[0]
    if stem.startswith("row_results_"):
        return stem[len("row_results_") :]
    return stem


def _build_jobs(configs: list[str] | None = None) -> list[dict[str, str]]:
    selected_configs = {config.strip() for config in (configs or []) if config.strip()}
    payload_names = sorted(name for name in os.listdir(PAYLOAD_DIR) if _dataset_name_from_payload(name))
    jobs: list[dict[str, str]] = []

    for payload_name in payload_names:
        dataset_name = _dataset_name_from_payload(payload_name)
        if not dataset_name:
            continue

        config_name = _config_name_from_dataset_name(dataset_name)
        if selected_configs and config_name not in selected_configs:
            continue

        dataset_path = os.path.join(DATASET_DIR, dataset_name)
        if not os.path.exists(dataset_path):
            print(f"Skipping payload without matching dataset: {payload_name}")
            continue

        output_dir = os.path.join(OUTPUT_ROOT, os.path.splitext(dataset_name)[0])
        jobs.append(
            {
                "payload_name": payload_name,
                "dataset_path": dataset_path,
                "output_dir": output_dir,
            }
        )

    return jobs


JOBS = _build_jobs(CONFIGS)

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
                "--parallel",
                "8",
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
print(f"Selected configs: {CONFIGS}")
if not JOBS:
    print(
        "No matching payload/dataset pairs were found. "
        f"Checked payloads in: {PAYLOAD_DIR} and datasets in: {DATASET_DIR}"
    )
for job in JOBS:
    print(f"- {job['dataset_path']}")
