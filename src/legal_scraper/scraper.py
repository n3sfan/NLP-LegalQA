"""High-level scraper that searches, fetches, and saves legal documents."""

from __future__ import annotations

import json
from pathlib import Path

from bs4 import BeautifulSoup

from legal_scraper.client import LegalDocumentClient


class LegalDocumentScraper:
    """Orchestrates searching and downloading legal documents."""

    def __init__(self, output_dir: str | Path = "data") -> None:
        self.client = LegalDocumentClient()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def search(self, keywords: str, **kwargs) -> list[dict]:
        """Search and return the list of document entries."""
        resp = self.client.search(keywords, **kwargs)
        return resp["data"]["docs"]

    def get_summary(self, doc_guid: str) -> dict:
        """Fetch the summary/metadata tab for a document."""
        return self.client.get_detail(doc_guid, tab="tomtat")["data"]

    def get_content(self, doc_guid: str) -> dict:
        """Fetch the full content tab for a document."""
        return self.client.get_detail(doc_guid, tab="noidung")["data"]

    @staticmethod
    def html_to_text(html: str) -> str:
        """Strip HTML tags and return clean text with line breaks preserved."""
        soup = BeautifulSoup(html, "html.parser")
        # Insert newlines before block-level elements so get_text() preserves structure
        for tag in soup.find_all(["p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
            tag.insert_before("\n")
        return soup.get_text()

    def save_document(self, doc_guid: str) -> Path:
        """Download a document's full content and save as .txt + .json metadata."""
        detail = self.get_content(doc_guid)
        safe_name = doc_guid

        # Save plain text content
        txt_path = self.output_dir / f"{safe_name}.txt"
        txt_path.write_text(self.html_to_text(detail.get("docContent", "")), encoding="utf-8")

        # Save metadata as JSON (everything except the large HTML blob)
        meta = {k: v for k, v in detail.items() if k != "docContent"}
        json_path = self.output_dir / f"{safe_name}.json"
        json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        return txt_path

    def scrape(self, keywords: str, max_docs: int | None = None, **search_kwargs) -> list[Path]:
        """Search for documents and save them all. Returns list of saved file paths."""
        docs = self.search(keywords, **search_kwargs)
        if max_docs is not None:
            docs = docs[:max_docs]

        saved = []
        for doc in docs:
            guid = doc["docGUId"]
            clear_name = doc.get("docNameClear", doc.get("docName", guid))
            print(f"  Downloading: {clear_name}")
            path = self.save_document(guid)
            saved.append(path)
            print(f"  Saved: {path}")

        return saved
