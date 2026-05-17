import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Any

@dataclass
class EvalConfig:
    dataset_path: str = "eval_results_reranker/row_results_reranker.csv"
    output_dir: str = "eval_results_qa_reranker/"
    payload_dir: str = "offline_payloads/"
    prompt_template_name: str = "prompt_qa_0shot.md"
    print_every: int = 5
    backend_type: str = "vllm"
    n_voters: int = 1
    models: List[str] = field(default_factory=lambda: ["Qwen/Qwen3-4B"])
    api_key: str = "vllm-secret-key"
    base_port: int = 8080
    neo4j_uri: str = "neo4j+ssc://nguyenhoangquan.com:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "Neoneo4j"
    neo4j_database: str = "neo4j"
    limit: Optional[int] = None
    start_index: int = 0
    batch_size: int = 10
    top_k: int = 5
    chk_recall: bool = False

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

def is_relevant(retrieved_uid: str, reference: str) -> bool:
    """Return True if `retrieved_uid` shares a prefix with `reference`."""
    return str(retrieved_uid).startswith(str(reference))

def get_payload_path(dataset_path: str, payload_dir: str) -> Path:
    p = Path(payload_dir)
    # If the user provided the full path to the jsonl file directly
    if p.suffix == ".jsonl" or p.is_file():
        return p
    dataset_name = Path(dataset_path).stem
    if dataset_name.startswith("row_results"):
        suffix = dataset_name.removeprefix("row_results").strip("_")
        payload_name = "row_results_payload"
        if suffix:
            payload_name = f"{payload_name}_{suffix}"
        return p / f"{payload_name}.jsonl"
    return p / f"{dataset_name}_payload.jsonl"

def get_val(row: Any, attr: str, default: str = "") -> str:
    """Safely get attribute from a row (namedtuple from itertuples)."""
    val = getattr(row, attr, default)
    return str(val).strip() if pd.notna(val) else default
