import os
from dotenv import load_dotenv

from legal_scraper.embedder import Neo4jEmbedder
from legal_scraper.reranker import VietnameseReranker

def main():
    load_dotenv()
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    print("[1] Connecting to DB & loading models...")
    embedder = Neo4jEmbedder(uri, user, password, database)
    reranker = VietnameseReranker()
    
    query = "không đội mũ bảo hiểm phạt bao nhiêu"
    
    print(f"\n[2] Executing Vector Search for: '{query}'")
    # k=30 per label, then we'll slice the overall top 30
    search_results = embedder.search(labels=["Point", "Clause", "Article"], query=query, k=30)
    
    search_results = search_results[:30]
    uids = [res.uid for res in search_results]
    print(f"    Found {len(search_results)} relevant nodes from vector search.")
    
    print("\n[3] Fetching Context Hierarchies from Neo4j...")
    hierarchy_map = embedder.fetch_node_hierarchy(uids)
    
    documents = []
    valid_uids = []
    vector_scores = []
    
    for res in search_results:
        if res.uid in hierarchy_map:
            documents.append(hierarchy_map[res.uid])
            valid_uids.append(res.uid)
            vector_scores.append(res.score)
            
    print(f"    Mapped hierarchies for {len(documents)} nodes.")
    
    print("\n[4] Reranking Results (Cross-Encoder)...")
    reranked_results = reranker.rerank(query, documents, top_k=30)
    
    print("\n=== TOP 30 RESULTS (RERANKED) ===")
    for rank, (idx, rerank_score) in enumerate(reranked_results):
        print(f"Rank {rank+1} | Reranker Score: {rerank_score:.4f} | Original Vector Score: {vector_scores[idx]:.4f}")
        print(f"UID: {valid_uids[idx]}")
        print("-" * 50)
        print(documents[idx])
        print("=" * 80)
        
    embedder.close()

if __name__ == "__main__":
    main()
