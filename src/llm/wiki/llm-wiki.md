# Legal QA LLM Wiki

Compact reference for `src/llm/`: how evaluation scripts fit together, what context builder to use, and the gotchas that tend to cause bad runs.

## Files

- `voter.py`: backend wrappers (`VLLMBackend`, `OpenRouterBackend`, `LlamaCppBackend`, `OllamaBackend`) plus `LegalVoter`.
- `eval_voter.py`: voter/classification evaluation and QA judge mode (`--mode eval_qa`).
- `eval_qa_online.py`: Neo4j-connected payload builder for offline QA inference.
- `eval_qa_offline.py`: network-free QA generation from payload JSONL; supports local vLLM ports and `--openrouter`.
- `eval_qa_utils.py`: shared `EvalConfig`, template loading, payload path resolution, UID relevance helper.
- `make_kaggle_cell.py`: bundles `src/llm/` logic into notebook-ready `cellcode.py`.

## Context Builders

Use `build_context_str_for_uids(embedder, uids, expand=False)` for QA payloads built from retrieval `row_results`.

- Defined in `legal_scraper.retrieval`.
- Input: final ranked UIDs from `retrieved_uids` in `row_results*.csv`.
- Output: prompt-ready `law_text` using the same context style as `retrieve_and_build_context().context_str`.
- Rebuilds the missing post-retrieval context steps from UIDs: hierarchy text, abolished/replaced tags, amendments, and optional sibling/child expansion.
- The prompt-ready value is usually:

```python
law_text = build_context_str_for_uids(embedder, uids, expand=False)
extra_info = ""
```

Use `fetch_law_texts(embedder, uids, batch_size)` only for voter/classification flows that need per-UID law text and metadata.


## QA Workflow

1. Build retrieval results, e.g. `scripts/eval_pipeline.py`, producing rows with `retrieved_uids`, references, and recall columns.
2. Generate offline payloads on a machine with Neo4j access:

```bash
uv run src/llm/eval_qa_online.py --dataset eval_results_v4/row_results_full_pipeline.csv
```

For an ablation folder, pass the folder and one payload JSONL will be written for each `row_results*.csv`:

```bash
uv run src/llm/eval_qa_online.py `
  --dataset eval_results/QA_Part2345/eval_results_pipeline_all_ablations_geminiflash_rerank_top_30 `
  --payload-dir offline_payloads/ `
  --top-k 15 
```

3. Run LLM inference from payloads:

```bash
python src/llm/eval_qa_offline.py \
  --dataset eval_results_v4/row_results_full_pipeline.csv \
  --models Qwen/Qwen3-4B \
  --base-port 8080
```

For OpenRouter:

```bash
OPENROUTER_API_KEY=... python src/llm/eval_qa_offline.py \
  --dataset eval_results_v4/row_results_full_pipeline.csv \
  --openrouter
```

4. Score generated answers with judge mode:

```bash
uv run .\src\llm\eval_voter.py `
--mode eval_qa `
--backend openrouter `
--input eval_results\thinh_eval_llm\finetune `
--dataset eval_results\QA_Part2\eval_results_pipeline_all_ablations_geminiflash_rerank_top_30 `
--gt-dataset qa_dataset/QA_Part2.csv `
--payload-dir .\offline_payloads `
--output .\eval_results\thinh_eval_llm\score\finetune `
--prompt-template .\src\llm\prompts\prompt_eval_qa_0shot.md `
--model google/gemini-2.5-flash `
--print-every 5
```

## Operational Notes

- Remote Neo4j should use `neo4j+ssc://...`.
- vLLM uses one server per model: `base_port + i`. QA defaults to `8080`; voter evaluation defaults to `8000`.
- `eval_qa_online.py` prefers the nearest available `recall@N` column where `N <= top_k`; if absent, it falls back to UID prefix matching.
- `eval_qa_online.py` accepts either one row-results CSV or a directory containing `row_results*.csv`.
- `get_payload_path(dataset_path, payload_dir)` maps `row_results_full_pipeline.csv` to `offline_payloads/row_results_full_pipeline_payload.jsonl` unless `payload_dir` is already a `.jsonl` path.
- Prompt templates may use `{question}`, `{law_text}`, `{extra_info}`, and sometimes `{top_k}`, `{ground_truth}`, or `{llm_answer}` depending on phase.
- Inference retries empty/error responses up to 3 times with a short backoff.
- Gemma-style model names trigger ChatML token remapping in offline QA/judge flows.

## Common Pitfalls

- `references` are UIDs, not expert answer text. For text comparison, merge the original QA dataset `answer` column via `--gt-dataset`.
- Preserve UTF-8 for Vietnamese legal text and CSV/JSONL files.
- Pandas may turn IDs into floats (`1` -> `1.0`). Use the existing `clean_id`/string normalization patterns when joining outputs, GT rows, and payloads.
- Most scripts are easiest to run from the repo root or `src/llm/`; imports are intentionally lightweight rather than packaged.
- Output CSVs are append/resume friendly. Use `--start-index` deliberately, and avoid rerunning from `0` unless you want headers/results reset.
