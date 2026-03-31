"""FastAPI server for QA reference annotation tool."""

from __future__ import annotations

import csv
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from annotate_qa.search import ArticleEntry, SearchIndex

# Config
APP_DIR = Path(__file__).parent
SRC_DIR = APP_DIR.parent.parent          # repo root
CSV_PATH = SRC_DIR / "qa_dataset" / "QA_NLP.csv"
PARSED_DIR = SRC_DIR / "data" / "parsed"


# In-memory state
_rows: list[dict] = []       # [{"id", "question", "answer", "reference"}, ...]
_index: SearchIndex = SearchIndex()

# Data loading / saving
def load_csv() -> None:
    global _rows
    _rows = []
    if not CSV_PATH.exists():
        return
    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                _rows.append({
                    "id": row.get("id", "").strip(),
                    "question": row.get("question", "").strip(),
                    "answer": row.get("answer", "").strip(),
                    "reference": row.get("reference", "").strip(),
                })
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV: {e}")


def save_csv() -> None:
    fieldnames = ["id", "question", "answer", "reference"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_rows)


# Pydantic models
class UpdateRefRequest(BaseModel):
    reference: str


class RowUpdateRequest(BaseModel):
    rows: list[dict]


# FastAPI app
app = FastAPI(title="QA Reference Annotation Tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    load_csv()
    _index.build(PARSED_DIR)


@app.get("/")
def root() -> FileResponse:
    static_path = APP_DIR / "static" / "index.html"
    if static_path.exists():
        return FileResponse(str(static_path))
    raise HTTPException(status_code=404, detail="index.html not found")


# CSV rows
@app.get("/api/rows")
def get_rows() -> list[dict]:
    return _rows


@app.get("/api/rows/{row_id}")
def get_row(row_id: str) -> dict:
    for row in _rows:
        if row["id"] == str(row_id):
            return row
    raise HTTPException(status_code=404, detail="Row not found")


@app.post("/api/rows/{row_id}")
def update_row(row_id: str, body: UpdateRefRequest) -> dict:
    for row in _rows:
        if row["id"] == str(row_id):
            row["reference"] = body.reference.strip()
            return row
    raise HTTPException(status_code=404, detail="Row not found")


@app.post("/api/save")
def api_save(body: RowUpdateRequest) -> dict:
    global _rows
    _rows = body.rows
    save_csv()
    return {"saved": len(_rows)}


# Search 
@app.get("/api/search")
def api_search(q: str = "") -> dict:
    articles, clauses, points = _index.search_all(q)
    return {
        "articles": [
            {
                "type": "article",
                "doc_identity": e.doc_identity,
                "doc_name": e.doc_name,
                "article_num": e.article_num,
                "title": e.title,
                "uid": e.uid,
            }
            for e in articles
        ],
        "clauses": [
            {
                "type": "clause",
                "doc_identity": c.doc_identity,
                "article_num": c.article_num,
                "clause_num": c.clause_num,
                "content": c.content,
                "uid": c.uid,
            }
            for c in clauses
        ],
        "points": [
            {
                "type": "point",
                "doc_identity": p.doc_identity,
                "article_num": p.article_num,
                "clause_num": p.clause_num,
                "point_letter": p.point_letter,
                "content": p.content,
                "uid": p.uid,
            }
            for p in points
        ],
    }


@app.get("/api/article")
def get_article_children(doc_identity: str, article_num: int) -> dict:
    """Return clauses and points for a given article."""
    clauses, points = _index.get_article_children(doc_identity, article_num)
    return {
        "doc_identity": doc_identity,
        "article_num": article_num,
        "article_uid": f"{doc_identity}::article::{article_num}",
        "clauses": [
            {
                "clause_num": c.clause_num,
                "content": c.content,
                "uid": c.uid,
            }
            for c in clauses
        ],
        "points": [
            {
                "clause_num": p.clause_num,
                "point_letter": p.point_letter,
                "content": p.content,
                "uid": p.uid,
            }
            for p in points
        ],
    }

# CLI entry point
def main() -> None:
    load_csv()
    _index.build(PARSED_DIR)
    total_cl = sum(len(v) for v in _index._clauses.values())
    total_pt = sum(len(v) for v in _index._points.values())
    print(f"Loaded SearchIndex: {len(_index.entries)} articles, {total_cl} clauses, {total_pt} points")
    print(f"Loaded {len(_rows)} CSV rows")
    print(f"Serving at http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
