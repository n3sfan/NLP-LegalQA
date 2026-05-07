# Legal QA: LLM & Evaluation Knowledge Base

This wiki document serves as the "source of truth" for AI agents and developers working on the `src/llm/` directory. It documents the core architecture, design decisions, and common pitfalls to avoid.

## 1. Directory Overview

- **`voter.py`**: Contains backend abstractions (`VLLMBackend`, `LlamaCppBackend`, `OllamaBackend`) and the `LegalVoter` orchestrator.
- **`eval_voter.py`**: The main evaluation script for the "Voter" phase. It handles data fetching from Neo4j and calculates metrics (Recall, Precision, MRR).
- **`eval_qa.py`**: Specialised script for the "Answer Generation" phase. It uses retrieved law context to generate natural language answers.
- **`eval_qa_online.py`**: Prepares "payload" files by fetching law context from Neo4j.
- **`eval_qa_offline.py`**: Executes LLM inference using pre-fetched payloads, requiring no network access.

## 2. Core Implementation Decisions

### 2.1. Hierarchical Law Fetching (`fetch_law_texts`)
**Function**: `fetch_law_texts(embedder, uids, batch_size)` in `eval_voter.py`.
- **Decision**: Law retrieval must NOT be a simple node fetch. Vietnamese legal documents follow a strict hierarchy: **Document -> Article -> Clause -> Point**.
- **Logic**: This function fetches the target UID and its parents (Article/Document) to build a human-readable text. It avoids redundant headers when merging multiple UIDs.
- **Dynamic Context**: It also retrieves document metadata (effective dates) and **amendments** (via `AMENDS` relationships).
- **Short Names**: The system automatically extracts document types (e.g., "Nghị định", "Luật") and pairs them with IDs for concise headers (e.g., "Nghị định 100/2019/NĐ-CP").
- **Important**: Use this for tasks requiring law grounding. It returns a 4-tuple: `(uid_to_text, nodes_metadata, merged_text, extra_info)`.

### 2.2. Multi-Model Backend Logic
**Class**: `VLLMBackend` in `voter.py`.
- **Backend**: Uses `langchain_openai.ChatOpenAI`. It is compatible with both `vLLM` and `llama.cpp` servers.
- **Port Incrementing**: To evaluate multiple models in parallel, the scripts use a `base_port + i` convention.
    - *Example*: Model 0 → Port 8080, Model 1 → Port 8081.
- **Mistake Avoidance**: When adding new model support, ensure the port logic matches this pattern so that separate server instances are correctly addressed.

### 2.3. Asynchronous Execution
- **Parallelism**: Both `LegalVoter` and `eval_qa.py` heavily use `asyncio.gather` to query models in parallel.
- **Decision**: For QA generation, all models should be queried concurrently for the same question to minimize evaluation time.

### 2.4. Template Management
- **Context Manager**: `_swap_prompt_template` in `eval_voter.py` allows temporary overrides of the default prompt files.
- **Loading**: `load_template(filename)` in `eval_qa.py` searches current and parent directories for `.md` files.

### 2.5. Metadata & Amendment Injection
**Utility**: `Neo4jEmbedder.format_amends` and `Neo4jEmbedder.format_uid_vn`.
- **Amendment Retrieval**: The system fetches amendments for both ancestor and descendant nodes (e.g., if an Article is requested, it finds amends for its Clauses/Points too).
- **Vietnamese Legal Formatting**: UIDs are automatically formatted into standard Vietnamese citations (e.g., `168/2024/NĐ-CP::article::52::clause::8` -> `Khoản 8 Điều 52 168/2024/NĐ-CP`) before injection.
- **Prompt Integration**: Amendments and effective dates are bundled into an `extra_info` string and injected via the `{extra_info}` placeholder in prompt templates.

## 3. Data Structures

- **`VoteResult`**: Returned by `LegalVoter.vote()`. Contains:
    - `verdict`: Boolean (Majority vote).
    - `votes`: List of individual boolean results.
    - `models`: List of model names used.
    - `raw_responses`: The full text returned by each LLM.
- **`extra_info`**: A pre-formatted string containing:
    - **Danh sách văn bản**: List of governing documents with their effective dates.
    - **Thông tin sửa đổi**: Formatted list of amendments (if any) using Vietnamese citation style.
- **UID Format**: `[DocID]::article::[N]::clause::[M]`.
- **Vietnamese Format**: `Điểm [P] Khoản [M] Điều [N] [DocID]`.
- **CSV Columns**:
    - `references`: Gold standard UIDs (ground truth).
    - `retrieved_uids`: UIDs returned by the search engine.
- **Neo4j URI**: Remote connections **must** use the `+ssc` protocol (e.g., `neo4j+ssc://...`).
- **Base Ports**: 
    - `eval_voter`: `8000`
    - `eval_qa`: `8080`

## 4. Common Pitfalls (Where Agents Make Mistakes)

1.  **Relative Imports**: The scripts are designed to be run from inside `src/llm/`. 
    - *Correct*: `from voter import ...`
    - *Incorrect*: `from src.llm.voter import ...` (unless the package is installed or the root is in PYTHONPATH).
2.  **Expert Answers vs. References**:
    - In the reranker results CSV, the `references` column contains UIDs.
    - The actual text of the expert answer is found in the original `qa_dataset/QA_NLP.csv` in the `answer` column.
    - **Mistake**: Using the UID as the expert answer text for text-to-text comparison.
3.  **Encoding**: Vietnamese text MUST be handled with `encoding="utf-8"`. Failure to do so will corrupt the legal articles and LLM outputs.
4.  **Incremental Saving**: Both evaluation scripts append to CSVs row-by-row (`open(..., 'a')`). This allows resuming from crashes using `--start-index`.

## 5. Usage Patterns

### Asking a Model (QA)
```python
# Standard pattern in eval_qa.py
tasks = [ask_model(backend, model_name) for backend, model_name in zip(backends, models)]
results = await asyncio.gather(*tasks)
```

### Fetching Law Text
```python
# Standard pattern in eval_voter.py
# The 3rd and 4th return values are pre-formatted context strings
_, _, law_text, extra_info = fetch_law_texts(embedder, uids, batch_size=batch_size)

# The prompt template should contain both {law_text} and {extra_info}
prompt = template.format(question=q, law_text=law_text, extra_info=extra_info)
```

## 6. Offline Evaluation Workflow

To decouple network-heavy law fetching from GPU-heavy LLM inference:

1. **Online Phase (Data Preparation)**:
   Run `eval_qa_online.py` on a machine with Neo4j access.
   ```bash
   python eval_qa_online.py --dataset eval_results_v2/row_results_decomposition.csv
   ```
   This generates a `.jsonl` file in `offline_payloads/`.

2. **Offline Phase (Inference)**:
   Transfer the payload to a GPU machine and run `eval_qa_offline.py`.
   ```bash
   python eval_qa_offline.py --models Qwen/Qwen2-7B-Instruct --base-port 8080
   ```
