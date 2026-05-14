# LLM Module

## Overview

This module provides **3-LLM majority-vote classification** for legal QA relevance, and evaluation tooling for measuring voter accuracy against a ground-truth RAG retrieval dataset.

```
src/llm/
├── voter.py                    # 3-LLM majority vote classifier (Ollama + vLLM backends)
├── eval_voter.py               # Evaluates voter on row_results.csv (RAG output)
├── prompt_classify.md          # Default classification prompt (few-shot)
├── prompt_classify_zero_shot.md # Zero-shot alternative prompt
└── README.md                   # This file
```

---

## voter.py — Majority Vote Classifier

### What it does

Given a `(question, law_text)` pair, runs **N concurrent LLM calls** and returns a majority-vote verdict (`Có` = relevant / `Không` = irrelevant).

Two backends are available:

| Backend | Speed | GPU | Notes |
|---|---|---|---|
| `OllamaBackend` | Slower (one call at a time per model) | Optional | Works on CPU too; good for dev |
| `VLLMBackend` | Fast (PagedAttention, batching) | Required | Recommended for evaluation |

### Key classes

| Class | Role |
|---|---|
| `OllamaBackend` | Calls a local Ollama server via `langchain-ollama` (JSON mode, default: `llama3.2`) |
| `VLLMBackend` | Calls a vLLM OpenAI-compatible server via `langchain-openai` |
| `VoterAgent` | Wraps one backend, sends prompt, parses `Có`/`Không` response |
| `LegalVoter` | Runs N agents concurrently via `asyncio.gather`, takes majority |
| `VoteResult` | Dataclass: `verdict: bool`, `votes: list[bool]`, `models`, `raw_responses`, `all_agreed` |

### Prompt template

`prompt_classify.md` is loaded by `load_prompt_template()`. Override at runtime via `_swap_prompt_template()` context manager (used by `eval_voter.py`).

Expected LLM response: `{"answer": "Có"}` or `{"answer": "Không"}` (JSON), with keyword fallback (`\b(có|yes)\b` → True, `\b(không|no)\b` → False).

### Usage — Ollama backend

```python
from llm.voter import LegalVoter, OllamaBackend

backends = [OllamaBackend(model="qwen3:4b") for _ in range(3)]
voter = LegalVoter(backends=backends, model_names=["qwen3:4b"]*3)

result = await voter.vote(
    question="Lái xe uống rượu phạt bao nhiêu?",
    law_text="Điều 6. Phạt tiền từ 18-20 triệu đồng...",
)
print(result.verdict)  # True (Có) or False (Không)
```

Or the convenience entry point:

```python
from llm.voter import majority_vote
result = await majority_vote(question, law_text, models=["qwen3:4b"], backend="ollama")
```

### Usage — vLLM backend

The vLLM server must already be running (see [vllm_colab_notebook.ipynb](../vllm_colab_notebook.ipynb)):

```bash
# In Colab (cell from vllm_colab_notebook.ipynb):
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --model tomng/nanbeige4.1:3b \
    --trust-remote-code --dtype half \
    --port 8000 --host 0.0.0.0 --api-key vllm-secret-key \
    --gpu-memory-utilization 0.85
```

Then in Python:

```python
from llm.voter import LegalVoter, VLLMBackend

backends = [
    VLLMBackend(model="Qwen/Qwen3-4B",      base_url="http://localhost:8000/v1", api_key="vllm-secret-key"),
    VLLMBackend(model="tomng/nanbeige4.1:3b", base_url="http://localhost:8000/v1", api_key="vllm-secret-key"),
]
voter = LegalVoter(backends=backends, model_names=["Qwen/Qwen3-4B", "tomng/nanbeige4.1:3b"])

result = await voter.vote(question, law_text)
```

Convenience entry point:

```python
from llm.voter import majority_vote
result = await majority_vote(
    question, law_text,
    models=["Qwen/Qwen3-4B"],
    backend="vllm",
    base_url="http://localhost:8000/v1",
    api_key="vllm-secret-key",
)
```

### vLLM server startup (Colab)

See `../vllm_colab_notebook.ipynb` — Section 5 launches the server with multiple models:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --model tomng/nanbeige4.1:3b \
    --trust-remote-code --dtype half \
    --port 8000 --host 0.0.0.0 --api-key vllm-secret-key \
    --gpu-memory-utilization 0.85
    # Add --quantization fp8 to save ~40% VRAM
```

---

## eval_voter.py — Voter Evaluation on RAG Results

### What it does

Reads `row_results.csv` (output of `scripts/eval_rag.py`), fetches the **law text for each top-K retrieved UID from Neo4j**, calls the voter on each, and computes **accuracy / precision / recall** against ground-truth references.

### Input: `row_results.csv` columns

| Column | Description |
|---|---|
| `id` | Question ID |
| `question` | Vietnamese legal question |
| `retrieved_uids` | Semicolon-separated list of retrieved node UIDs (up to 30, sorted by cosine score) |
| `references` | Semicolon-separated ground-truth reference UIDs |

### Law text assembly (Node → Prompt format)

For each retrieved UID, the script fetches the node + parent nodes from Neo4j and assembles law text in standard Vietnamese legal document format:

```
# Article node
Nghị định 168 Về trật tự an toàn giao thông đường bộ
Điều 6. Xử phạt, trừ điểm giấy phép lái xe...
<article content>

# Clause node  (parent Article fetched automatically)
Nghị định 168 Về trật tự an toàn giao thông đường bộ
Điều 6. Xử phạt, trừ điểm giấy phép lái xe...
4. <clause content>

# Point node  (parent Article + Clause fetched automatically)
Nghị định 168 Về trật tự an toàn giao thông đường bộ
Điều 6. Xử phạt, trừ điểm giấy phép lái xe...
4. <clause content>
d) <point content>
```

- `Điều N.` prefix is reconstructed from the UID (`{doc}::article::{N}`)
- `N.` clause prefix and `x)` point prefix are reconstructed from UIDs
- `doc_name` (e.g. "Nghị định 168 Về trật tự...") is fetched from the `Document` node via `doc_identity`
- Content duplicated at parent level (same text in article + clause) is auto-skipped

### Neo4j fetch strategy (per question)

1. Collect all needed UIDs: retrieved UIDs + parent Article UIDs + parent Clause UIDs
2. One batched `MATCH (n) WHERE n.uid IN $uids` query
3. One batched `MATCH (d:Document) WHERE d.doc_identity IN $identities` query for doc names
4. Pass parents + doc_name to `_build_law_text()` for each UID

### Ground-truth matching

Same logic as `scripts/eval_rag.py`: `is_relevant(retrieved_uid, reference) = retrieved_uid.startswith(reference)`.

### Metrics

| Metric | Definition |
|---|---|
| **accuracy** | `(verdict == is_correct) / total` |
| **precision** | `TP / (TP + FP)` — voter said Có AND uid matches reference |
| **recall** | `TP / (TP + FN)` — uid matches reference AND voter said Có |

Where: TP = verdict=True + correct, FP = verdict=True + not correct, FN = verdict=False + correct.

### Output files

| File | Contents |
|---|---|
| `row_voter_results.csv` | Per-UID detail: `id, question, uid, label, is_correct, verdict, law_text_preview` |
| `voter_metrics_summary.csv` | Macro-average: `accuracy, precision, recall, total_questions, total_uids` |

### CLI usage

```bash
# Ollama (default) — 2 models × 3 voters
uv run python -m src.llm.eval_voter \
    --input eval_results/row_results.csv \
    --output eval_results/ \
    --uri "neo4j+ssc://nguyenhoangquan.com:7687" \
    --user neo4j --password "Neoneo4j" --database neo4j \
    --models qwen3:4b qwen3:4b qwen3:4b \
    --backend ollama

# vLLM (GPU-accelerated) — specify model IDs served by vLLM
uv run python -m src.llm.eval_voter \
    --input eval_results/row_results.csv \
    --output eval_results/ \
    --uri "neo4j+ssc://nguyenhoangquan.com:7687" \
    --user neo4j --password "Neoneo4j" --database neo4j \
    --models Qwen/Qwen3-4B tomng/nanbeige4.1:3b \
    --backend vllm \
    --vllm-url "http://localhost:8000/v1" \
    --vllm-api-key "vllm-secret-key"

# Zero-shot prompt
--prompt-template src/llm/prompt_classify_zero_shot.md

# Change top-K (default 5)
--top-k 3

# Env var fallbacks (also respected)
export NEO4J_URI="neo4j+ssc://..."
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="..."
export NEO4J_DATABASE="neo4j"
```

### Key helper functions

| Function | Purpose |
|---|---|
| `_parent_article_uid(uid)` | Extract `{doc}::article::{N}` from clause/point UID |
| `_parent_clause_uid(uid)` | Extract `{doc}::article::{N}::clause::{M}` from point UID |
| `_prefix_article_title(title, uid)` | Prepend `Điều N.` to article title |
| `_prefix_clause_content(content, uid)` | Prepend `N.` to clause content |
| `_prefix_point_content(content, uid)` | Prepend `x)` to point content |
| `_build_law_text(node, parent_article, parent_clause, doc_name)` | Assemble full law text for prompt |
| `_fetch_nodes_sync(driver, uids)` | Batched node fetch by UID |
| `_fetch_documents_sync(driver, identities)` | Batched doc_name fetch by doc_identity |
| `is_relevant(retrieved_uid, reference)` | Same as eval_rag.py: startswith check |
| `compute_metrics(rows)` | Compute accuracy / precision / recall |
| `_swap_prompt_template(path)` | Context manager: temporarily override prompt template |

### Neo4j node properties used

| Property | Node | Purpose |
|---|---|---|
| `uid` | All content nodes | Primary key, used for lookup and UID-based parent extraction |
| `title` | Article | Article title text |
| `content` | Article, Clause, Point | Body text |
| `doc_identity` | Article, Clause, Point | Foreign key to Document |
| `doc_name` | Document | Full law name for prompt header |
| `labels(n)[0]` | All | Node label (Article, Clause, Point) |
