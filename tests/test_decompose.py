from dotenv import load_dotenv
load_dotenv()
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")

from legal_scraper.embedder import Neo4jEmbedder

e = Neo4jEmbedder(
    uri=os.environ["NEO4J_URI"],
    user=os.environ["NEO4J_USER"],
    password=os.environ["NEO4J_PASSWORD"],
    database=os.environ.get("NEO4J_DATABASE", "neo4j"),
    openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
)

# Bước 1: decompose
decomp = e.decompose_query_debug("So sánh mức phạt vượt đèn đỏ của xe ô tô và xe máy?")

print("=== REASONING (CoT) ===")
print(decomp.reasoning)
print("\n=== SUB-QUERIES ===")
for i, sq in enumerate(decomp.sub_queries):
    print(f"  [{i}] {sq.get('query', sq)}")
print(f"\nSuccess: {decomp.success}")

if decomp.sub_queries:
    results = e.multi_search(decomp.sub_queries, k=3)
    print("\n=== SEARCH RESULTS ===")
    for idx, hits in results.items():
        sq = decomp.sub_queries[idx]
        print(f"\nSQ[{idx}]: {sq.get('query', sq)}")
        for r in hits:
            print(f"  uid={r.uid} | label={r.label} | score={r.score:.4f}")

e.close()
