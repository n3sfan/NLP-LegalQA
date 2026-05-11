"""CLI entry point for the legal document scraper."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add src/ to path so imports like 'from legal_scraper...' work when running via python -m
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from legal_scraper.parser import LegalDocumentParser
from legal_scraper.scraper import LegalDocumentScraper
from legal_scraper.neo4j_importer import Neo4jImporter
from legal_scraper.embedder import Neo4jEmbedder


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    sys.stdout.reconfigure(encoding="utf-8")
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
    p_import.add_argument("-a", "--amends-input", default="data/amends", help="Directory with amends JSON files")
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

    # --- search-rerank ---
    p_vsr = sub.add_parser("search-rerank", help="Perform vector search and rerank results with cross-encoder")
    p_vsr.add_argument("--uri", default=os.getenv("NEO4J_URI"), required=not os.getenv("NEO4J_URI"), help="Neo4j connection URI (e.g. neo4j+ssc://host:7687)")
    p_vsr.add_argument("--user", default=os.getenv("NEO4J_USER"), required=not os.getenv("NEO4J_USER"))
    p_vsr.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"), required=not os.getenv("NEO4J_PASSWORD"))
    p_vsr.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    p_vsr.add_argument("--query", "-q", required=True, help="Vietnamese text query")
    p_vsr.add_argument(
        "--labels",
        nargs="+",
        default=["Article", "Clause", "Point"],
        choices=["Article", "Clause", "Point"],
        help="Node labels to search (default: Article Clause Point)",
    )
    p_vsr.add_argument("--fetch-k", type=int, default=30, help="Initial top-k candidates per label from vector search (default: 30)")
    p_vsr.add_argument("--top-k", type=int, default=5, help="Final top-k results to display after reranking (default: 5)")
    p_vsr.add_argument("--full", action="store_true", help="Show full content instead of truncated")

    # --- create-fulltext-index ---
    p_fti = sub.add_parser("create-fulltext-index", help="Create fulltext (BM25) indexes on Neo4j for hybrid search")
    p_fti.add_argument("--uri", default=os.getenv("NEO4J_URI"), required=not os.getenv("NEO4J_URI"))
    p_fti.add_argument("--user", default=os.getenv("NEO4J_USER"), required=not os.getenv("NEO4J_USER"))
    p_fti.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"), required=not os.getenv("NEO4J_PASSWORD"))
    p_fti.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))

    # --- query (full pipeline) ---
    p_query = sub.add_parser("query", help="Full retrieval pipeline: decompose → search → aggregate → rerank with amends")
    p_query.add_argument("--uri", default=os.getenv("NEO4J_URI"), required=not os.getenv("NEO4J_URI"))
    p_query.add_argument("--user", default=os.getenv("NEO4J_USER"), required=not os.getenv("NEO4J_USER"))
    p_query.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"), required=not os.getenv("NEO4J_PASSWORD"))
    p_query.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    p_query.add_argument("--query", "-q", required=True, help="Vietnamese text query")
    p_query.add_argument(
        "--labels",
        nargs="+",
        default=["Article", "Clause", "Point"],
        choices=["Article", "Clause", "Point"],
    )
    p_query.add_argument("--decompose", dest="decompose", action="store_true", help="Enable query decomposition (default)")
    p_query.add_argument("--no-decompose", dest="decompose", action="store_false", help="Disable query decomposition")
    p_query.set_defaults(decompose=True)
    p_query.add_argument("--hybrid", dest="hybrid", action="store_true", help="Enable hybrid search: vector + BM25 keyword (default)")
    p_query.add_argument("--no-hybrid", dest="hybrid", action="store_false", help="Disable hybrid search, use vector only")
    p_query.set_defaults(hybrid=True)
    p_query.add_argument("--aggregate", choices=["rrf", "borda", "max"], default="rrf", help="Aggregation strategy (default: rrf)")
    p_query.add_argument("--fetch-k", type=int, default=30, help="Initial candidates per sub-query/label (default: 30)")
    p_query.add_argument("--rerank-top", type=int, default=15, help="Number of candidates to rerank with cross-encoder (default: 15)")
    p_query.add_argument("--top-k", type=int, default=5, help="Final results for QA context after reranking (default: 5)")
    p_query.add_argument("--full", action="store_true", help="Show full content")
    p_query.add_argument("--no-hierarchy", dest="hierarchy", action="store_false", help="Skip hierarchy fetching (default: fetch)")
    p_query.set_defaults(hierarchy=True)

    # --- chat (interactive multi-turn REPL) ---
    p_chat = sub.add_parser("chat", help="Interactive multi-turn chat with conversation history")
    p_chat.add_argument("--uri", default=os.getenv("NEO4J_URI"), required=not os.getenv("NEO4J_URI"))
    p_chat.add_argument("--user", default=os.getenv("NEO4J_USER"), required=not os.getenv("NEO4J_USER"))
    p_chat.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"), required=not os.getenv("NEO4J_PASSWORD"))
    p_chat.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    p_chat.add_argument(
        "--labels",
        nargs="+",
        default=["Article", "Clause", "Point"],
        choices=["Article", "Clause", "Point"],
    )
    p_chat.add_argument("--fetch-k", type=int, default=30)
    p_chat.add_argument("--rerank-top", type=int, default=15, help="Candidates to rerank with cross-encoder (default: 15)")
    p_chat.add_argument("--top-k", type=int, default=8, help="Final results for QA context (default: 8)")
    p_chat.add_argument("--max-history", type=int, default=10, help="Max conversation turns to keep (default: 10)")
    p_chat.add_argument("--provider", choices=["local", "openrouter"], default=None, help="LLM provider override (default: from LLM_PROVIDER env)")
    p_chat.add_argument("--no-decompose", dest="decompose", action="store_false")
    p_chat.set_defaults(decompose=True)
    p_chat.add_argument("--no-hybrid", dest="hybrid", action="store_false")
    p_chat.set_defaults(hybrid=True)
    p_chat.add_argument("--aggregate", choices=["rrf", "borda", "max"], default="rrf")
    p_chat.add_argument("--expand", dest="expand", action="store_true", help="Enable Article/Clause children + sibling Points expansion")
    p_chat.add_argument("--no-expand", dest="expand", action="store_false", help="Disable expansion (default)")
    p_chat.set_defaults(expand=False)

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
            print(f"Importing parsed directory: {args.input}")
            summary = importer.import_parsed_directory(
                Path(args.input),
                fail_fast=args.fail_fast,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            
            amends_dir = Path(args.amends_input)
            if amends_dir.exists() and amends_dir.is_dir():
                print(f"Importing amends directory: {args.amends_input}")
                amends_summary = importer.import_amends_directory(
                    amends_dir,
                    fail_fast=args.fail_fast,
                )
                print(json.dumps(amends_summary, ensure_ascii=False, indent=2))
            else:
                print(f"Amends directory not found or not a directory: {args.amends_input}")

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

    elif args.command == "create-fulltext-index":
        embedder = Neo4jEmbedder(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        )
        try:
            print("Creating fulltext indexes for hybrid search...")
            embedder.create_fulltext_indexes()
            print("Done. Fulltext indexes are ready.")
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

    elif args.command == "search-rerank":
        from legal_scraper.reranker import VietnameseReranker
        embedder = Neo4jEmbedder(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        )
        try:
            print(f"Executing Vector Search for: '{args.query}'...")
            search_results = embedder.search(args.labels, args.query, k=args.fetch_k)
            search_results = search_results[:args.fetch_k]
            
            if not search_results:
                print("No results found.")
                return

            uids = [res.uid for res in search_results]
            print(f"Found {len(search_results)} relevant nodes. Fetching context hierarchies...")
            hierarchy_map = embedder.fetch_node_hierarchy(uids)

            documents = []
            valid_results = []
            
            for res in search_results:
                if res.uid in hierarchy_map:
                    documents.append(hierarchy_map[res.uid])
                    valid_results.append(res)
            
            print(f"Mapped hierarchies for {len(documents)} nodes. Loading reranker...")
            reranker = VietnameseReranker()
            
            print("Reranking results...")
            reranked = reranker.rerank(args.query, documents, top_k=args.top_k)

            top_k_uids = [valid_results[idx].uid for (idx, _) in reranked]
            amends_map = embedder.fetch_amends(top_k_uids)

            print("\n=== TOP RESULTS (RERANKED) ===")
            for rank, (idx, rerank_score) in enumerate(reranked, 1):
                orig_res = valid_results[idx]
                content = documents[idx]

                if not args.full:
                    if len(content) > 300:
                        content_display = content[:300] + f"\n  ... ({len(content)} chars total)"
                    else:
                        content_display = content
                else:
                    content_display = content

                print(f"\n[{rank}] [{orig_res.label}] rerank_score={rerank_score:.4f}  (vec_score={orig_res.score:.4f})  uid={orig_res.uid}")
                print(f"  ---\n  {content_display}\n  ---")
                
                amends_text = Neo4jEmbedder.format_amends(amends_map, [orig_res.uid])
                if amends_text:
                    print(amends_text)
        finally:
            embedder.close()
        return

    elif args.command == "query":
        from legal_scraper.retrieval import aggregate_search_results, fetch_context_for_results
        from legal_scraper.reranker import VietnameseReranker

        embedder = Neo4jEmbedder(uri=args.uri, user=args.user, password=args.password, database=args.database)
        try:
            from legal_scraper.router import QueryRouter
            from legal_scraper.generator import AnswerGenerator
            
            # Step 0: Routing
            print(f"Routing query: '{args.query}'...")
            router = QueryRouter()
            intent = router.route(args.query)
            print(f"Intent classified as: {intent}")
            
            generator = AnswerGenerator()
            
            if intent == "reject":
                print("\n=== TRẢ LỜI ===")
                print("Xin lỗi, tôi là một chatbot pháp luật giao thông đường bộ Việt Nam. Câu hỏi của bạn nằm ngoài phạm vi tư vấn của tôi.")
                return
                
            elif intent == "direct_answer":
                print("\nGenerating direct answer...")
                ans = generator.generate_direct_answer(args.query)
                print("\n=== TRẢ LỜI ===")
                print(ans)
                return

            # Step 1: Retrieval (decompose or single search)
            if args.decompose:
                from legal_scraper.query_parser import QueryDecomposer
                
                print("Decomposing query...")
                decomposer = QueryDecomposer()
                
                try:
                    sub_queries = decomposer.decompose(args.query)
                    # Always include original query as fallback for BM25 keyword coverage
                    sub_queries.append({"query": args.query})
                    print(f"Decomposed into {len(sub_queries)} sub-queries (incl. original):")
                    for i, sq in enumerate(sub_queries):
                        print(f"  {i+1}. {sq['query']}")
                except Exception as e:
                    print(f"Decomposition failed: {e}. Falling back to single query.")
                    sub_queries = [{"query": args.query}]
                
                raw_results = embedder.multi_search(sub_queries, k=args.fetch_k, hybrid=args.hybrid)
                search_results = aggregate_search_results(raw_results, strategy=args.aggregate)[:args.fetch_k]
                # Use the rewritten/decomposed query for reranking (exclude original to keep legal precision)
                rerank_query = " ".join([sq["query"] for sq in sub_queries[:-1]])
            else:
                search_fn = embedder.hybrid_search if args.hybrid else embedder.search
                search_results = search_fn(args.labels, args.query, k=args.fetch_k)[:args.fetch_k]
                rerank_query = args.query  # no rewrite available, use original

            if not search_results:
                print("No results found.")
                return

            # Step 2: Fetch context (hierarchy or basic)
            context_map = fetch_context_for_results(embedder, search_results, include_hierarchy=args.hierarchy)

            final_results = search_results
            rerank_scores = {}
            rerank_pool = args.rerank_top if hasattr(args, 'rerank_top') else 15
            if rerank_pool > 0 and len(search_results) > 0:
                # Rerank a wider pool, then take top_k for QA context
                pool_size = min(rerank_pool, len(search_results))
                documents = [context_map.get((r.uid, r.label), "") for r in search_results[:pool_size]]
                reranker = VietnameseReranker()
                reranked_indices = reranker.rerank(rerank_query, documents, top_k=pool_size)
                # Take top_k from the reranked pool for final results
                top_indices = reranked_indices[:args.top_k]
                final_results = [search_results[idx] for idx, _ in top_indices]
                # Build score mapping: original index -> rerank score
                rerank_scores = {idx: score for idx, score in reranked_indices}

            # Step 4: Fetch amends for top results
            top_k_uids = [r.uid for r in final_results[:args.top_k]]
            amends_map = embedder.fetch_amends(top_k_uids)

            # Step 5: Display results
            print(f"\nQuery: {args.query}")
            print(f"Decompose: {args.decompose}, Hybrid: {args.hybrid}, Aggregate: {args.aggregate}, Rerank: {bool(args.top_k > 0)}")
            print(f"\nTop {len(final_results)} results:\n")

            score_label = "agg_score" if args.decompose else "vec_score"

            for rank, r in enumerate(final_results, 1):
                ctx = context_map.get((r.uid, r.label), "[no content]")
                if not args.full and len(ctx) > 300:
                    ctx_display = ctx[:300] + f"\n... ({len(ctx)} chars total)"
                else:
                    ctx_display = ctx

                # Get original index in search_results for rerank score lookup
                orig_idx = search_results.index(r)
                if orig_idx in rerank_scores:
                    score = rerank_scores[orig_idx]
                    print(f"[{rank}] [{r.label}] rerank_score={score:.4f}  ({score_label}={r.score:.4f})  uid={r.uid}")
                else:
                    print(f"[{rank}] [{r.label}] {score_label}={r.score:.4f}  uid={r.uid}")
                print(f"  ---\n  {ctx_display}\n  ---")

                amends_text = Neo4jEmbedder.format_amends(amends_map, [r.uid])
                if amends_text:
                    print(amends_text)

            # Step 6: Generate final RAG answer
            print("\nGenerating final answer from retrieved contexts...")
            context_blocks = []
            for r in final_results[:args.top_k]:
                ctx = context_map.get((r.uid, r.label), "")
                amends = amends_map.get(r.uid, [])
                if amends:
                    amend_str = "\n".join([f"Đã được sửa đổi/bổ sung: {a['amending_content']}" for a in amends])
                    ctx = f"{ctx}\n\n[LƯU Ý - NỘI DUNG SỬA ĐỔI]:\n{amend_str}"
                context_blocks.append(ctx)
            
            context_str = "\n\n---\n\n".join(context_blocks)
            final_answer = generator.generate_rag_answer(rerank_query, context_str)
            print("\n=== TRẢ LỜI ===")
            print(final_answer)
        finally:
            embedder.close()
        return

    elif args.command == "chat":
        from legal_scraper.retrieval import aggregate_search_results, fetch_context_for_results
        from legal_scraper.reranker import VietnameseReranker
        from legal_scraper.query_rewriter import QueryRewriter
        from legal_scraper.router import QueryRouter
        from legal_scraper.generator import AnswerGenerator

        embedder = Neo4jEmbedder(uri=args.uri, user=args.user, password=args.password, database=args.database)
        try:
            # Initialise components
            print("[*] Initializing components...")
            t_init = time.time()
            rewriter = QueryRewriter()  # uses LLM_PROVIDER env
            router = QueryRouter()
            generator = AnswerGenerator()
            reranker = VietnameseReranker()
            print(f"[*] Components ready ({time.time() - t_init:.1f}s)")
            provider = os.getenv("LLM_PROVIDER", "local")
            print(f"[*] LLM Provider: {provider}")
            print(f"[*] Model:    {rewriter.llm.model_name}")
            print(f"[*] Base URL: {rewriter.llm.openai_api_base}")

            chat_history: list[dict] = []

            print("\n" + "=" * 60)
            print("  Legal QA Chat (multi-turn) — type 'exit' or 'quit' to stop")
            print("  Type '/clear' to reset conversation history")
            print("=" * 60 + "\n")

            while True:
                try:
                    user_input = input("[USER]: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    print("Goodbye!")
                    break
                if user_input.lower() == "/clear":
                    chat_history.clear()
                    print("[*] Conversation history cleared.\n")
                    continue

                # --- Step 0: Rewrite query using conversation history ---
                t0 = time.time()
                rewritten_query = rewriter.rewrite(chat_history, user_input)
                t_rewrite = time.time() - t0

                if rewritten_query != user_input:
                    print(f"[*] Rewritten: '{user_input}' → '{rewritten_query}' ({t_rewrite:.2f}s)")
                else:
                    print(f"[*] Query unchanged (no rewrite needed, {t_rewrite:.2f}s)")

                # --- Step 1: Route ---
                t1 = time.time()
                intent = router.route(rewritten_query)
                t_route = time.time() - t1
                print(f"[*] Route: {intent} ({t_route:.2f}s)")

                if intent == "reject":
                    answer = "Xin lỗi, tôi là chatbot pháp luật giao thông đường bộ Việt Nam. Câu hỏi của bạn nằm ngoài phạm vi tư vấn của tôi."
                    print(f"\n[ASSISTANT]: {answer}\n")
                    chat_history.append({"role": "user", "content": user_input})
                    chat_history.append({"role": "assistant", "content": answer})

                elif intent == "direct_answer":
                    t2 = time.time()
                    answer = generator.generate_direct_answer(rewritten_query)
                    t_gen = time.time() - t2
                    print(f"\n[ASSISTANT]: {answer}")
                    print(f"[*] Generation: {t_gen:.2f}s\n")
                    chat_history.append({"role": "user", "content": user_input})
                    chat_history.append({"role": "assistant", "content": answer})

                else:  # retrieve
                    # --- Step 2: Decompose ---
                    if args.decompose:
                        from legal_scraper.query_parser import QueryDecomposer
                        t2 = time.time()
                        decomposer = QueryDecomposer()
                        try:
                            sub_queries = decomposer.decompose(rewritten_query)
                            sub_queries.append({"query": rewritten_query})
                            t_decompose = time.time() - t2
                            print(f"[*] Decomposed into {len(sub_queries)} sub-queries ({t_decompose:.2f}s):")
                            for i, sq in enumerate(sub_queries):
                                print(f"    {i+1}. {sq['query']}")
                        except Exception as e:
                            print(f"[!] Decomposition failed: {e}. Using single query.")
                            sub_queries = [{"query": rewritten_query}]

                        raw_results = embedder.multi_search(sub_queries, k=args.fetch_k, hybrid=args.hybrid)
                        search_results = aggregate_search_results(raw_results, strategy=args.aggregate)[:args.fetch_k]
                        rerank_query = " ".join([sq["query"] for sq in sub_queries[:-1]])
                    else:
                        t2 = time.time()
                        search_fn = embedder.hybrid_search if args.hybrid else embedder.search
                        search_results = search_fn(args.labels, rewritten_query, k=args.fetch_k)[:args.fetch_k]
                        rerank_query = rewritten_query
                        print(f"[*] Search: {time.time() - t2:.2f}s")

                    if not search_results:
                        answer = "Không tìm thấy kết quả phù hợp trong cơ sở dữ liệu pháp luật."
                        print(f"\n[ASSISTANT]: {answer}\n")
                        chat_history.append({"role": "user", "content": user_input})
                        chat_history.append({"role": "assistant", "content": answer})
                        continue

                    # --- Step 3: Fetch context & rerank ---
                    t3 = time.time()
                    # Rerank a wider pool, then apply graph-based score adjustments
                    rerank_pool = min(args.rerank_top, len(search_results))
                    context_map = fetch_context_for_results(embedder, search_results[:rerank_pool], include_hierarchy=True)
                    documents = [context_map.get((r.uid, r.label), "") for r in search_results[:rerank_pool]]
                    reranked_indices = reranker.rerank(rerank_query, documents, top_k=rerank_pool)
                    t_rerank = time.time() - t3
                    print(f"[*] Context + Rerank: {t_rerank:.2f}s ({rerank_pool} candidates)")

                    # --- Step 3b: Graph-based score adjustments ---
                    t3b = time.time()
                    pool_uids = [search_results[idx].uid for idx, _ in reranked_indices]
                    
                    # Check abolished/replaced status
                    abolished_map = embedder.fetch_abolished_uids(pool_uids)
                    
                    # Get document effect dates for recency boost
                    doc_ids = list({uid.split("::")[0] for uid in pool_uids})
                    effect_dates = embedder.fetch_doc_effect_dates(doc_ids)
                    
                    from datetime import datetime, date
                    today = date.today()
                    
                    adjusted_indices = []
                    for idx, score in reranked_indices:
                        uid = search_results[idx].uid
                        doc_id = uid.split("::")[0]
                        
                        # Additive penalty for abolished/replaced provisions
                        # (multiplicative breaks for negative cross-encoder scores)
                        amend_types = abolished_map.get(uid, [])
                        if "bãi bỏ" in amend_types:
                            score -= 5.0  # heavy penalty
                        elif "thay thế" in amend_types:
                            score -= 3.0
                        
                        # Additive recency boost: newer documents score higher
                        eff_str = effect_dates.get(doc_id)
                        if eff_str:
                            try:
                                eff_date = datetime.strptime(eff_str, "%Y-%m-%d").date()
                                years_old = max(0, (today - eff_date).days) / 365.0
                                recency_bonus = max(0, 2.0 - 0.3 * years_old)  # +2 for brand new, 0 for 6+ years old
                                score += recency_bonus
                            except ValueError:
                                pass
                        
                        adjusted_indices.append((idx, score))
                    
                    # Re-sort by adjusted score and take top_k
                    adjusted_indices.sort(key=lambda x: x[1], reverse=True)
                    final_results = [search_results[idx] for idx, _ in adjusted_indices[:args.top_k]]
                    final_scores = adjusted_indices[:args.top_k]
                    
                    t_boost = time.time() - t3b
                    print(f"[*] Graph boost: {t_boost:.2f}s (abolished penalties + recency boost applied)")

                    # --- Step 4: Build context & generate ---
                    top_k_uids = [r.uid for r in final_results]
                    amends_map = embedder.fetch_amends(top_k_uids)
                    
                    # Expand context: sibling points + article/clause children
                    # Context expansion (disable with --no-expand for local LLMs)
                    siblings_map = {}
                    children_map = {}
                    if args.expand:
                        point_uids = [r.uid for r in final_results if r.label == "Point"]
                        siblings_map = embedder.fetch_sibling_points(point_uids) if point_uids else {}
                        parent_uids = [r.uid for r in final_results if r.label in ("Article", "Clause")]
                        children_map = embedder.fetch_children_context(parent_uids) if parent_uids else {}

                    # Debug: show retrieved results with hierarchy content
                    print(f"\n[DEBUG] Top {len(final_results)} results (after graph boost):")
                    for rank, (r, (_, score)) in enumerate(zip(final_results, final_scores), 1):
                        amend_count = len(amends_map.get(r.uid, []))
                        amend_tag = f" [+{amend_count} amends]" if amend_count else ""
                        abolished_types = abolished_map.get(r.uid, [])
                        abolished_tag = f" [{'|'.join(abolished_types)}]" if abolished_types else ""
                        uid_short = Neo4jEmbedder.format_uid_vn(r.uid)
                        print(f"\n  {rank}. [{r.label}] score={score:.4f} {uid_short}{amend_tag}{abolished_tag}")
                        # Print the hierarchy context (what the LLM sees for this node)
                        ctx = context_map.get((r.uid, r.label), "[no context]")
                        print(f"     [Hierarchy context]:")
                        for line in ctx.split("\n"):
                            print(f"       {line}")
                        # Print siblings if any
                        if r.uid in siblings_map:
                            print(f"     [Siblings added by --expand]:")
                            for line in siblings_map[r.uid].split("\n"):
                                print(f"       {line}")

                    if not args.expand:
                        print("\n[DEBUG] --expand OFF, skipping sibling/children fetch.")

                    context_blocks = []
                    for r in final_results:
                        ctx = context_map.get((r.uid, r.label), "")
                        
                        # Tag abolished provisions
                        abolished_types = abolished_map.get(r.uid, [])
                        if "bãi bỏ" in abolished_types:
                            ctx = f"[ĐÃ BỊ BÃI BỎ bởi văn bản mới hơn]\n{ctx}"
                        elif "thay thế" in abolished_types:
                            ctx = f"[ĐÃ BỊ THAY THẾ bởi văn bản mới hơn]\n{ctx}"
                        
                        # Append sibling points for Point nodes
                        if r.uid in siblings_map:
                            ctx += f"\n\n[Các điểm khác cùng khoản]:\n{siblings_map[r.uid]}"
                        
                        # Expand Article/Clause nodes with children content
                        if r.uid in children_map:
                            ctx += f"\n\n[Nội dung chi tiết]:\n{children_map[r.uid]}"
                        
                        amends = amends_map.get(r.uid, [])
                        if amends:
                            amend_str = "\n".join([f"Đã được sửa đổi/bổ sung: {a['amending_content']}" for a in amends])
                            ctx = f"{ctx}\n\n[LƯU Ý - NỘI DUNG SỬA ĐỔI]:\n{amend_str}"
                        context_blocks.append(ctx)
                    context_str = "\n\n---\n\n".join(context_blocks)

                    t4 = time.time()
                    answer = generator.generate_rag_answer(rerank_query, context_str)
                    t_gen = time.time() - t4

                    print(f"\n[ASSISTANT]: {answer}")
                    print(f"[*] Generation: {t_gen:.2f}s\n")

                    # Sanitize surrogates from LLM responses (OpenRouter sometimes returns them)
                    def _sanitize(text: str) -> str:
                        return text.encode("utf-8", errors="replace").decode("utf-8")
                    
                    clean_answer = _sanitize(answer)
                    chat_history.append({"role": "user", "content": user_input})
                    chat_history.append({"role": "assistant", "content": clean_answer})

                # Trim history to max_history turns (each turn = 2 messages)
                max_msgs = args.max_history * 2
                if len(chat_history) > max_msgs:
                    chat_history = chat_history[-max_msgs:]

        finally:
            embedder.close()
        return

if __name__ == "__main__":
    main()
