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

# Extract amendment relationships from parsed documents
uv run legal-scraper amend -i data/parsed/ -o data/amends/

# Extract amendments from a single document
uv run legal-scraper amend -d e97b70fe-0672-4800-3bfb-39d6be8eb58d -i data/parsed/ -o data/amends/
```

## Project Structure

```
src/legal_scraper/
  client.py       - Low-level HTTP client wrapping the phapluat.gov.vn API (3 endpoints)
  scraper.py      - High-level scraper: search, fetch, parse HTML, save .txt + .json
  parser.py       - Metadata + content parsers producing Neo4j-ready structured JSON
  models.py        - Dataclasses for all graph node types (Document, Chapter, Article, etc.)
  amend_extractor.py - Amendment extraction using NuExtract API
  cli.py          - CLI entry point (search, scrape, get, parse, amend commands)
data/            - Scraped document output (plain text + JSON metadata)
data/parsed/     - Parsed structured JSON output (one file per document)
data/amends/     - Extracted amendment relationships JSON output
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
