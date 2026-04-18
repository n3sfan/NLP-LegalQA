# CLAUDE.md

## Project Overview

Vietnamese legal document scraper and QA toolkit. Scrapes legal documents from `phapluat.gov.vn` API and parses them into structured entities for Neo4j graph RAG.

## Setup

```bash
uv sync
```

## Usage

```bash
# Search for documents
uv run legal-scraper search "Luật" -n 10 --from "01/01/2026" --to "15/02/2026"

# Download documents matching a search
uv run legal-scraper scrape "Luật" -n 5 -o data/

# Download a single document by GUID
uv run legal-scraper get <GUID> -o data/

# Parse downloaded documents into Neo4j-ready JSON
uv run legal-scraper parse -i data/ -o data/parsed/

# Parse a single document by GUID stem
uv run legal-scraper parse -d e97b70fe-0672-4800-3bfb-39d6be8eb58d

# Import parsed JSON into Neo4j
uv run legal-scraper import-neo4j \
  -i data/parsed \
  --uri "neo4j+ssc://<host>:7687" \
  --user neo4j \
  --password <password> \
  --database neo4j

# Generate vector embeddings for Neo4j nodes
uv run legal-scraper embed \
  --uri "neo4j+ssc://<host>:7687" \
  --user neo4j \
  --password <password> \
  --node-labels Article Clause Point

# Search vector indexes and return ranked results with content
uv run legal-scraper vector-search \
  --uri "neo4j+ssc://<host>:7687" \
  --user neo4j \
  --password <password> \
  --query "không đội mũ bảo hiểm phạt bao nhiêu" \
  --labels Article Clause Point --k 5

# Start the QA annotation web tool (requires annotate extra)
uv run annotate
```

## Running Tests

```bash
uv run pytest                    # all tests
uv run pytest tests/test_neo4j_importer.py  # single file
```

## Project Structure

```
src/legal_scraper/
  client.py           - Low-level HTTP client wrapping the phapluat.gov.vn API (3 endpoints)
  scraper.py          - High-level scraper: search, fetch, parse HTML, save .txt + .json
  parser.py           - Metadata + content parsers producing Neo4j-ready structured JSON
  models.py           - Dataclasses for all graph node types (Document, Chapter, Article, etc.)
  neo4j_importer.py   - MERGE-based importer for parsed JSON into Neo4j (single-pass, idempotent)
  embedder.py         - Neo4j vector embedding via LangChain + Neo4jVector; also handles vector search
                        and hierarchy fetching (variable-length path queries)
  reranker.py         - Vietnamese cross-encoder reranker (AITeamVN/Vietnamese_Reranker, FP16)
  cli.py              - CLI entry point (search, scrape, get, parse, import-neo4j, embed, vector-search)
src/annotate_qa/       - Standalone FastAPI server for QA reference annotation
  server.py           - FastAPI app: serves index.html, exposes /api/search and /api/article endpoints
  search.py           - In-memory SearchIndex (ArticleEntry, ClauseEntry, PointEntry)
  static/index.html   - Frontend for browsing and annotating QA references
data/                - Scraped document output (plain text + JSON metadata)
data/parsed/         - Parsed structured JSON output (one file per document)
data/amends/         - Extracted amendment relationships JSON output
tests/
  test_neo4j_importer.py  - Unit tests for UID builders, constraint statements, payload loading
  test_embedder.py         - Unit tests for Neo4jEmbedder init and lazy driver
  test_hierarchy_fetch.py  - Integration script: fetch_node_hierarchy via variable-length path query
  test_reranker.py         - Integration script: vector search → fetch hierarchy → cross-encoder rerank
```

## API Endpoints

- `POST /api/legal-documents` — search with keywords, date range, filters
- `GET /api/legal-documents/detail?docGUId=...&tabName=tomtat` — document summary/metadata
- `GET /api/legal-documents/detail?docGUId=...&tabName=noidung` — full document content (HTML)

## Knowledge Graph Schema

### Metadata Graph
- **Nodes**: Document, DocumentGroup, DocumentType, EffectStatus, Organization, Signer, Field
- **Relationships**: BELONGS_TO_GROUP, HAS_TYPE, HAS_STATUS, ISSUED_BY, SIGNED_BY, IN_FIELD, RELATED_TO

### Content Graph (Document Hierarchy)
- **Nodes**: Document → Part → Chapter → Section → Article → Clause → Point
- **Relationships**: HAS_PART, HAS_CHAPTER, HAS_SECTION, HAS_ARTICLE, HAS_CLAUSE, HAS_POINT

| Vietnamese | English | Format |
|---|---|---|
| Phần | Part | `Phần thứ nhất. TITLE` |
| Chương | Chapter | `Chương I. TITLE` |
| Mục | Section | `Mục 1. TITLE` |
| Điều | Article | `Điều 1. Title` |
| Khoản | Clause | `1. content` |
| Điểm | Point | `a) content` |

## Key Conventions

- Python 3.12+, managed with `uv`
- Build system: hatchling with `src/` layout
- Documents saved as `{docGUId}.txt` (content) + `{docGUId}.json` (metadata)
- Uses `docGUId` (UUID) for filenames to avoid collisions with duplicate `docIdentity` values

## Neo4j Import

### Import command
```bash
uv run legal-scraper import-neo4j \
  -i data/parsed \
  --uri "neo4j+ssc://<host>:7687" \
  --user neo4j \
  --password <password> \
  --database neo4j
```

The `neo4j+ssc://` scheme is required for self-hosted Neo4j instances using self-signed SSL certificates.

### Node identity strategy
Content hierarchy nodes use document-scoped composite UIDs to avoid collisions across documents that reuse the same numbering (Điều 1, Khoản 1, điểm a, etc.):

| Label | UID format |
|---|---|
| Document | `doc_identity` (e.g. `56/2024/QH15`) |
| Part | `{doc_identity}::part::{number}` |
| Chapter | `{doc_identity}::chapter::{number}` |
| Section | `{doc_identity}::section::{number}` |
| Article | `{doc_identity}::article::{number}` |
| Clause | `{doc_identity}::article::{parent_article}::clause::{number}` |
| Point | `{doc_identity}::article::{parent_article}::clause::{parent_clause}::point::{letter}` |

Metadata nodes (DocumentGroup, DocumentType, EffectStatus, Organization, Signer, Field) use `id` as their key.

### Relationship directions
- `Document → Article` via `HAS_ARTICLE`
- `Document → Part` via `HAS_PART`
- `Document → Chapter` via `HAS_CHAPTER` (direct, when no Part exists)
- `Part → Chapter` via `HAS_CHAPTER`
- `Chapter → Section` via `HAS_SECTION`
- `Chapter → Article` via `HAS_ARTICLE` (when no Section exists)
- `Section → Article` via `HAS_ARTICLE`
- `Article → Clause` via `HAS_CLAUSE`
- `Clause → Point` via `HAS_POINT`
- `Document → Document` via `RELATED_TO` (cross-references; stubs created if target not yet imported)

### Data quirks (known)
- The parsed JSON for `118/2025/QH15` contains 2 duplicate Clause entries (same UID). The importer correctly `MERGE`-deduplicates them — only one survives in Neo4j. This is a source-data artifact, not a bug.
- Some clause numbering in source documents is sequential within an article rather than reflecting the original document's clause numbers. The importer preserves whatever numbers are in the parsed JSON.

## Vector Search

Embeddings generated using `bkai-foundation-models/vietnamese-bi-encoder` (PhoBERT-base-v2, 768-dim, cosine-normalized, local GPU inference on CUDA).

Vector indexes exist on `Article`, `Clause`, and `Point` nodes (property: `embedding`).

### Search via LangChain Neo4jVector

Uses Neo4j's native vector index — no GDS plugin required:

```python
from legal_scraper.embedder import Neo4jEmbedder

e = Neo4jEmbedder(
    uri="neo4j+ssc://nguyenhoangquan.com:7687",
    user="neo4j",
    password="Neoneo4j",
    database="neo4j",
)
results = e.search(
    labels=["Article", "Clause", "Point"],
    query="không đội mũ bảo hiểm phạt bao nhiêu",
    k=5,
)
for r in results:
    print(f"[{r.score:.4f}] [{r.label}] {r.uid}")
e.close()
```

Returns `List[SearchResult(uid, label, score)]`, sorted by cosine similarity descending. Pass a single label string instead of a list to search just one index.

Vector indexes are created per node label (e.g. `Article_embedding_index`). Embeddings are idempotent — re-runs skip nodes that already have an embedding.
