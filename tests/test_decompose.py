from dotenv import load_dotenv
load_dotenv()
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")

from legal_scraper.embedder import Neo4jEmbedder, SearchResult
from legal_scraper.reranker import VietnameseReranker

e = Neo4jEmbedder(
    uri=os.environ["NEO4J_URI"],
    user=os.environ["NEO4J_USER"],
    password=os.environ["NEO4J_PASSWORD"],
    database=os.environ.get("NEO4J_DATABASE", "neo4j"),
    openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
)

query = "So sánh mức phạt vượt đèn đỏ của xe ô tô và xe máy?"
decomp = e.decompose_query_debug(query)

print("=== CoT REASONING ===")
print(decomp.reasoning)
print("\n=== SUB-QUERIES ===")
for i, sq in enumerate(decomp.sub_queries):
    print(f"  [{i}] {sq.get('query', sq)}")
print(f"\nSuccess: {decomp.success}")

if decomp.sub_queries:
    # Vector search each sub-query independently
    results = e.multi_search(decomp.sub_queries, k=5)

    # Fetch node content for all UIDs
    all_uids, all_labels, uid_to_idx = [], [], {}
    for idx, hits in results.items():
        for r in hits:
            all_uids.append(r.uid)
            all_labels.append(r.label)
            uid_to_idx[(r.uid, r.label)] = idx

    node_contents = e.fetch_nodes(all_uids, list(set(all_labels)))

    # Rerank per sub-query
    reranker = VietnameseReranker(device="cpu")

    print("\n=== SCORE COMPARISON ===")
    for idx, hits in results.items():
        sq = decomp.sub_queries[idx]
        query_text = sq.get("query", sq)

        docs, doc_map = [], []
        for r in hits:
            key = (r.uid, r.label)
            content = node_contents.get(key, {}).get("content", "")
            title = node_contents.get(key, {}).get("title") or ""
            text = (f"{title}\n{content}" if title else content).strip()
            docs.append(text)
            doc_map.append(r)

        if not docs:
            continue

        reranked = reranker.rerank(query_text, docs, top_k=len(docs), batch_size=4)

        print(f"\nSQ[{idx}]: {query_text}")
        for orig_idx, rerank_score in reranked:
            r = doc_map[orig_idx]
            vec_rank = sorted(hits, key=lambda x: x.score, reverse=True).index(r) + 1
            rerank_rank = reranked.index((orig_idx, rerank_score)) + 1
            change = vec_rank - rerank_rank
            print(f"  #{vec_rank:>2}->#{rerank_rank:<2} ({change:>+3}) | vec={r.score:.4f} re={rerank_score:.4f} | {r.uid}")

e.close()
