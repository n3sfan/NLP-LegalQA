# NLP-LegalQA

NLP-LegalQA is a Vietnamese legal question answering toolkit. It combines legal document scraping, parsing, Neo4j graph import, vector retrieval, reranking, LLM-based answer generation, evaluation scripts, and a Streamlit chat UI.

The project is built for experiments around Vietnamese legal QA and retrieval-augmented generation over structured legal documents.

## Features

- Scrape Vietnamese legal documents from `phapluat.gov.vn`.
- Parse downloaded documents into structured JSON suitable for graph import.
- Import legal articles, clauses, and points into Neo4j.
- Generate embeddings and run vector, full-text, hybrid, and reranked retrieval.
- Run Legal QA chat through a FastAPI backend and Streamlit UI.
- Evaluate retrieval and generation pipelines with scripts for baseline RAG, rerankers, fine-tuned LLMs, BERTScore, ROUGE, and merged evaluation outputs.
- Provide annotation tooling for QA references.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/legal_scraper/` | CLI, scraper, parser, Neo4j importer, retrieval, API, and QA pipeline code. |
| `src/llm/` | LLM-related helpers. |
| `src/annotate_qa/` | FastAPI annotation tool for QA reference data. |
| `scripts/` | Dataset generation, evaluation, fine-tuning, and result-processing scripts. |
| `streamlit_ui/` | Streamlit frontend for the Legal QA API. |
| `tests/` | Unit and integration-style tests for parsing, retrieval, embedding, and graph import behavior. |
| `notebook/` | Exploratory notebooks and supporting resources. |
| `qa_dataset/` | QA dataset artifacts used by the experiments. |

## Requirements

- Python 3.12+
- Neo4j for graph import, retrieval, and QA pipeline runs
- `uv` or `pip`
- Optional local or hosted OpenAI-compatible LLM endpoint
- Optional OpenRouter API key for hosted model experiments

## Setup

With `uv`:

```bash
uv sync
```

Or with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For UI dependencies:

```bash
pip install -e '.[ui]'
```

Copy the environment template:

```bash
cp .env.example .env
```

Configure Neo4j and model provider values in `.env`.

## CLI Examples

Search legal documents:

```bash
legal-scraper search "giao thông đường bộ" -n 10
```

Scrape documents:

```bash
legal-scraper scrape "giao thông đường bộ" -n 20 -o data
```

Parse downloaded documents:

```bash
legal-scraper parse -i data -o data/parsed
```

Import parsed documents into Neo4j:

```bash
legal-scraper import-neo4j -i data/parsed
```

Run vector search:

```bash
legal-scraper vector-search -q "quy định về tốc độ xe máy" --k 5
```

Run the full QA pipeline:

```bash
legal-scraper query -q "Người điều khiển xe máy cần giấy tờ gì?"
```

## API and UI

Start the FastAPI backend:

```bash
uv run uvicorn legal_scraper.api:app --host 0.0.0.0 --port 8000 --reload
```

Run the Streamlit UI:

```bash
cd streamlit_ui
streamlit run app.py
```

The UI expects the API backend at `http://localhost:8000`.

## Tests

```bash
pytest
```

Some tests and pipeline commands require local data files, Neo4j, model weights, or API credentials. Keep secrets in `.env`; do not commit local credentials or generated payloads.

## Status

This is an active research/course project for Vietnamese legal QA. The codebase is organized for experimentation and evaluation, not yet as a packaged production service.

## License

MIT