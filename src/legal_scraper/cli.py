"""CLI entry point for the legal document scraper."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from legal_scraper.parser import LegalDocumentParser
from legal_scraper.scraper import LegalDocumentScraper
from legal_scraper.neo4j_importer import Neo4jImporter
from legal_scraper.embedder import Neo4jEmbedder


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
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

    # --- import-neo4j ---
    p_import = sub.add_parser("import-neo4j", help="Import parsed JSON into Neo4j")
    p_import.add_argument("-i", "--input", default="data/parsed", help="Directory with parsed JSON files")
    p_import.add_argument("--uri", default=os.getenv("NEO4J_URI"), required=not os.getenv("NEO4J_URI"), help="Neo4j URI (e.g. neo4j://localhost:7687)")
    p_import.add_argument("--user", default=os.getenv("NEO4J_USER"), required=not os.getenv("NEO4J_USER"), help="Neo4j username")
    p_import.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"), required=not os.getenv("NEO4J_PASSWORD"), help="Neo4j password")
    p_import.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"), help="Neo4j database name (default: neo4j)")
    p_import.add_argument("--fail-fast", action="store_true", help="Stop on first import error")

    # --- embed ---
    p_embed = sub.add_parser("embed", help="Generate vector embeddings for Neo4j nodes")
    p_embed.add_argument("--uri", default=os.getenv("NEO4J_URI"), required=not os.getenv("NEO4J_URI"), help="Neo4j connection URI (e.g. neo4j+ssc://host:7687)")
    p_embed.add_argument("--user", default=os.getenv("NEO4J_USER"), required=not os.getenv("NEO4J_USER"))
    p_embed.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"), required=not os.getenv("NEO4J_PASSWORD"))
    p_embed.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    p_embed.add_argument(
        "--node-labels",
        nargs="+",
        default=["Article"],
        choices=["Article", "Clause", "Point"],
        help="Node labels to embed (default: Article)",
    )
    p_embed.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding (default: 32)",
    )

    # --- vector-search ---
    p_vs = sub.add_parser("vector-search", help="Search Neo4j vector indexes and return ranked results with content")
    p_vs.add_argument("--uri", default=os.getenv("NEO4J_URI"), required=not os.getenv("NEO4J_URI"), help="Neo4j connection URI (e.g. neo4j+ssc://host:7687)")
    p_vs.add_argument("--user", default=os.getenv("NEO4J_USER"), required=not os.getenv("NEO4J_USER"))
    p_vs.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"), required=not os.getenv("NEO4J_PASSWORD"))
    p_vs.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    p_vs.add_argument("--query", "-q", required=True, help="Vietnamese text query")
    p_vs.add_argument(
        "--labels",
        nargs="+",
        default=["Article"],
        choices=["Article", "Clause", "Point"],
        help="Node labels to search (default: Article)",
    )
    p_vs.add_argument("--k", type=int, default=5, help="Top-k results per label (default: 5)")
    p_vs.add_argument("--full", action="store_true", help="Show full content instead of truncated")

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

   
    elif args.command == "import-neo4j":
        importer = Neo4jImporter(
            args.uri,
            args.user,
            args.password,
            args.database,
        )
        try:
            importer.ensure_constraints()
            summary = importer.import_parsed_directory(
                Path(args.input),
                fail_fast=args.fail_fast,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        finally:
            importer.close()

    elif args.command == "embed":
        embedder = Neo4jEmbedder(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        )
        try:
            print(f"Embedding {args.node_labels} nodes...")
            embedder.embed_label(args.node_labels, batch_size=args.batch_size)
            print("  Done.")
        finally:
            embedder.close()
        return

    elif args.command == "vector-search":
        embedder = Neo4jEmbedder(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        )
        try:
            results = embedder.search(args.labels, args.query, k=args.k)
            if not results:
                print("No results found.")
                return

            uids_by_label: dict[str, list[str]] = {}
            for r in results:
                uids_by_label.setdefault(r.label, []).append(r.uid)

            all_labels = list(uids_by_label.keys())
            all_uids = [r.uid for r in results]
            node_data = embedder.fetch_nodes(all_uids, all_labels)

            for rank, r in enumerate(results, 1):
                key = (r.uid, r.label)
                data = node_data.get(key, {"content": "[not found]", "title": None})

                title = data["title"]
                content = data["content"]

                if args.full:
                    content_display = content
                else:
                    if len(content) > 300:
                        content_display = content[:300] + f"\n  ... ({len(content)} chars total)"
                    else:
                        content_display = content

                print(f"[{rank}] [{r.label}] score={r.score:.4f}  uid={r.uid}")
                if r.label == "Article" and title:
                    print(f"  Title: {title}")
                print(f"  ---\n  {content_display}\n  ---")
        finally:
            embedder.close()
        return
