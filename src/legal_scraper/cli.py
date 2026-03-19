"""CLI entry point for the legal document scraper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from legal_scraper.parser import LegalDocumentParser
from legal_scraper.scraper import LegalDocumentScraper
from legal_scraper.amend_extractor import AmendExtractor


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

    # --- amend ---
    p_amend = sub.add_parser("amend", help="Extract amendment relationships from parsed documents")
    p_amend.add_argument("-i", "--input", default="data/parsed", help="Input directory with parsed JSON files")
    p_amend.add_argument("-o", "--output", default="data/amends", help="Output directory for amends JSON")
    p_amend.add_argument("-d", "--doc-id", help="Extract amendments from single document by stem")
    p_amend.add_argument("--api-key", help="NuExtract API key (optional, uses default if not provided)")
    p_amend.add_argument("--project-id", help="NuExtract project ID (optional, uses default if not provided)")

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
        doc_parser = LegalDocumentParser()
        input_dir = Path(args.input)
        output_dir = Path(args.output)

        if args.doc_id:
            output_dir.mkdir(parents=True, exist_ok=True)
            result = doc_parser.parse_document(args.doc_id, input_dir)
            out_path = output_dir / f"{args.doc_id}.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Parsed: {out_path}")
        else:
            saved = doc_parser.parse_directory(input_dir, output_dir)
            print(f"Parsed {len(saved)} documents to {output_dir}/")

    elif args.command == "amend":
        doc_parser = LegalDocumentParser()
        # Use provided values or None (which will use defaults in AmendExtractor)
        api_key = args.api_key if args.api_key else None
        project_id = args.project_id if args.project_id else None
        extractor = AmendExtractor(api_key=api_key, project_id=project_id)
        input_dir = Path(args.input)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.doc_id:
            # Extract from single document - read parsed JSON directly
            parsed_path = input_dir / f"{args.doc_id}.json"
            data = json.loads(parsed_path.read_text(encoding="utf-8"))
            doc_identity = data.get("nodes", {}).get("document", {}).get("doc_identity", args.doc_id)
            preamble = data.get("preamble", "")
            articles = data.get("nodes", {}).get("articles", [])
            clauses = data.get("nodes", {}).get("clauses", [])
            points = data.get("nodes", {}).get("points", [])

            print(f"Extracting amendments from {doc_identity}...")
            with open("cli_debug.txt", "w", encoding="utf-8") as f:
                f.write(f"preamble: {preamble[:200] if preamble else 'EMPTY'}\n")
            amends = extractor.extract_from_articles(articles, doc_identity, clauses, points, preamble)

            output = {
                "doc_identity": doc_identity,
                "amends": amends
            }
            out_path = output_dir / f"{args.doc_id}_amends.json"
            out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Extracted {len(amends)} amendments: {out_path}")
        else:
            # Extract from all documents in directory
            stems = {p.stem for p in input_dir.glob("*.json")}
            total_amends = 0
            processed = 0

            for stem in sorted(stems):
                try:
                    # Read parsed JSON directly
                    parsed_path = input_dir / f"{stem}.json"
                    if not parsed_path.exists():
                        print(f"  Skipping {stem}: parsed file not found")
                        continue

                    data = json.loads(parsed_path.read_text(encoding="utf-8"))
                    doc_identity = data.get("nodes", {}).get("document", {}).get("doc_identity", stem)
                    preamble = data.get("preamble", "")
                    articles = data.get("nodes", {}).get("articles", [])
                    clauses = data.get("nodes", {}).get("clauses", [])
                    points = data.get("nodes", {}).get("points", [])

                    if not articles:
                        print(f"  Skipping {stem}: no articles found")
                        continue

                    amends = extractor.extract_from_articles(articles, doc_identity, clauses, points, preamble)
                    processed += 1

                    if amends:
                        output = {
                            "doc_identity": doc_identity,
                            "amends": amends
                        }
                        out_path = output_dir / f"{stem}_amends.json"
                        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
                        total_amends += len(amends)
                        print(f"  {doc_identity}: {len(amends)} amendments")
                except Exception as e:
                    print(f"  Skipping {stem}: {e}")

            print(f"\nDone. Extracted {total_amends} amendments from {processed} documents to {output_dir}/")
