"""CLI entry point for the legal document scraper."""

from __future__ import annotations

import argparse
import sys

from legal_scraper.parser import LegalDocumentParser
from legal_scraper.scraper import LegalDocumentScraper


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scrape Vietnamese legal documents from phapluat.gov.vn")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- search ---
    p_search = sub.add_parser("search", help="Search for documents and print results")
    p_search.add_argument("keywords", help="Search keywords")
    p_search.add_argument("-n", "--limit", type=int, default=10, help="Max results to show")
    p_search.add_argument("--from", dest="date_from", default="01/01/1945", help="Date from (dd/mm/yyyy)")
    p_search.add_argument("--to", dest="date_to", default="15/02/2026", help="Date to (dd/mm/yyyy)")

    # --- scrape ---
    p_scrape = sub.add_parser("scrape", help="Search and download documents")
    p_scrape.add_argument("keywords", help="Search keywords")
    p_scrape.add_argument("-n", "--limit", type=int, default=None, help="Max documents to download")
    p_scrape.add_argument("-o", "--output", default="data", help="Output directory")
    p_scrape.add_argument("--from", dest="date_from", default="01/01/1945", help="Date from (dd/mm/yyyy)")
    p_scrape.add_argument("--to", dest="date_to", default="15/02/2026", help="Date to (dd/mm/yyyy)")

    # --- get ---
    p_get = sub.add_parser("get", help="Download a single document by GUID")
    p_get.add_argument("guid", help="Document GUID")
    p_get.add_argument("-o", "--output", default="data", help="Output directory")

    # --- parse ---
    p_parse = sub.add_parser("parse", help="Parse documents into Neo4j-ready JSON")
    p_parse.add_argument("-i", "--input", default="data", help="Input directory with .json/.txt files")
    p_parse.add_argument("-o", "--output", default="data/parsed", help="Output directory for parsed JSON")
    p_parse.add_argument("-d", "--doc-id", help="Parse single document by stem (e.g. 59-2020-QH14)")

    args = parser.parse_args(argv)

    if args.command == "search":
        scraper = LegalDocumentScraper()
        docs = scraper.search(args.keywords, date_from=args.date_from, date_to=args.date_to, row_amount=args.limit)
        print(f"Found {len(docs)} documents:\n")
        for i, doc in enumerate(docs, 1):
            name = doc.get("docNameClear", doc.get("docName", ""))
            status = doc.get("effectStatusName", "")
            guid = doc["docGUId"]
            print(f"  {i}. [{status}] {name}")
            print(f"     GUID: {guid}")
            print()

    elif args.command == "scrape":
        scraper = LegalDocumentScraper(output_dir=args.output)
        print(f"Searching for '{args.keywords}'...")
        saved = scraper.scrape(args.keywords, max_docs=args.limit, date_from=args.date_from, date_to=args.date_to)
        print(f"\nDone. Saved {len(saved)} documents to {args.output}/")

    elif args.command == "get":
        scraper = LegalDocumentScraper(output_dir=args.output)
        path = scraper.save_document(args.guid)
        print(f"Saved: {path}")

    elif args.command == "parse":
        from pathlib import Path

        doc_parser = LegalDocumentParser()
        input_dir = Path(args.input)
        output_dir = Path(args.output)

        if args.doc_id:
            output_dir.mkdir(parents=True, exist_ok=True)
            result = doc_parser.parse_document(args.doc_id, input_dir)
            out_path = output_dir / f"{args.doc_id}.json"
            import json
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Parsed: {out_path}")
        else:
            saved = doc_parser.parse_directory(input_dir, output_dir)
            print(f"Parsed {len(saved)} documents to {output_dir}/")


if __name__ == "__main__":
    main()
