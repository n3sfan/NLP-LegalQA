"""Automated test for graph-enhanced chat pipeline."""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

from legal_scraper.embedder import Neo4jEmbedder
from legal_scraper.reranker import VietnameseReranker
from legal_scraper.retrieval import aggregate_search_results, fetch_context_for_results
from legal_scraper.query_parser import QueryDecomposer
from legal_scraper.generator import AnswerGenerator
from datetime import datetime, date

# Init
embedder = Neo4jEmbedder(
    uri=os.getenv("NEO4J_URI"), user=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD"), database=os.getenv("NEO4J_DATABASE", "neo4j")
)
reranker = VietnameseReranker(device="cpu")
decomposer = QueryDecomposer()
generator = AnswerGenerator()

QUERIES = [
    "vượt đèn đỏ xe máy phạt bao nhiêu",
    "uống rượu bia lái xe ô tô bị phạt gì",
    "đi ngược chiều đường một chiều xe máy",
]

RERANK_TOP = 20
TOP_K = 10

for q_idx, query in enumerate(QUERIES, 1):
    print(f"\n{'='*80}")
    print(f"TEST {q_idx}: {query}")
    print(f"{'='*80}")
    
    # Decompose
    try:
        sub_queries = decomposer.decompose(query)
        sub_queries.append({"query": query})
        print(f"[*] Decomposed into {len(sub_queries)} sub-queries")
    except Exception as e:
        print(f"[!] Decompose failed: {e}")
        sub_queries = [{"query": query}]
    
    # Search
    raw_results = embedder.multi_search(sub_queries, k=20, hybrid=True)
    search_results = aggregate_search_results(raw_results, strategy="rrf")[:20]
    print(f"[*] Retrieved {len(search_results)} results")
    
    if not search_results:
        print("[!] No results found")
        continue
    
    # Rerank
    rerank_pool = min(RERANK_TOP, len(search_results))
    context_map = fetch_context_for_results(embedder, search_results[:rerank_pool], include_hierarchy=True)
    documents = [context_map.get((r.uid, r.label), "") for r in search_results[:rerank_pool]]
    reranked_indices = reranker.rerank(query, documents, top_k=rerank_pool)
    
    # Post-retrieval heuristic re-ranking
    pool_uids = [search_results[idx].uid for idx, _ in reranked_indices]
    abolished_map = embedder.fetch_abolished_uids(pool_uids)
    doc_ids = list({uid.split("::")[0] for uid in pool_uids})
    effect_dates = embedder.fetch_doc_effect_dates(doc_ids)
    today = date.today()
    
    adjusted_indices = []
    for idx, score in reranked_indices:
        uid = search_results[idx].uid
        doc_id = uid.split("::")[0]
        amend_types = abolished_map.get(uid, [])
        if "bãi bỏ" in amend_types:
            score -= 5.0
        elif "thay thế" in amend_types:
            score -= 3.0
        eff_str = effect_dates.get(doc_id)
        if eff_str:
            try:
                eff_date = datetime.strptime(eff_str, "%Y-%m-%d").date()
                years_old = max(0, (today - eff_date).days) / 365.0
                score += max(0, 2.0 - 0.3 * years_old)
            except ValueError:
                pass
        adjusted_indices.append((idx, score))
    
    adjusted_indices.sort(key=lambda x: x[1], reverse=True)
    final_results = [search_results[idx] for idx, _ in adjusted_indices[:TOP_K]]
    final_scores = adjusted_indices[:TOP_K]
    
    # Show results
    print(f"\n[DEBUG] Top {len(final_results)} results (after heuristic re-ranking):")
    for rank, (r, (_, score)) in enumerate(zip(final_results, final_scores), 1):
        abolished_types = abolished_map.get(r.uid, [])
        abolished_tag = f" [{'|'.join(abolished_types)}]" if abolished_types else ""
        uid_short = Neo4jEmbedder.format_uid_vn(r.uid)
        doc_id = r.uid.split("::")[0]
        eff = effect_dates.get(doc_id, "N/A")
        print(f"  {rank}. [{r.label}] score={score:.4f} {uid_short}{abolished_tag} (eff={eff})")
    
    # Build context with graph enhancements
    top_k_uids = [r.uid for r in final_results]
    amends_map = embedder.fetch_amends(top_k_uids)
    point_uids = [r.uid for r in final_results if r.label == "Point"]
    siblings_map = embedder.fetch_sibling_points(point_uids) if point_uids else {}
    
    context_blocks = []
    for r in final_results:
        ctx = context_map.get((r.uid, r.label), "")
        abolished_types = abolished_map.get(r.uid, [])
        if "bãi bỏ" in abolished_types:
            ctx = f"[ĐÃ BỊ BÃI BỎ bởi văn bản mới hơn]\n{ctx}"
        elif "thay thế" in abolished_types:
            ctx = f"[ĐÃ BỊ THAY THẾ bởi văn bản mới hơn]\n{ctx}"
        if r.uid in siblings_map:
            ctx += f"\n\n[Các điểm khác cùng khoản]:\n{siblings_map[r.uid]}"
        amends = amends_map.get(r.uid, [])
        if amends:
            amend_str = "\n".join([f"Đã được sửa đổi/bổ sung: {a['amending_content']}" for a in amends])
            ctx = f"{ctx}\n\n[LƯU Ý - NỘI DUNG SỬA ĐỔI]:\n{amend_str}"
        context_blocks.append(ctx)
    context_str = "\n\n---\n\n".join(context_blocks)
    
    # Verify context has document headers
    has_headers = "[Văn bản:" in context_str
    has_dates = "Hiệu lực:" in context_str
    print(f"\n[CHECK] Context has doc headers: {has_headers}")
    print(f"[CHECK] Context has effect dates: {has_dates}")
    print(f"[CHECK] Context has abolished tags: {'ĐÃ BỊ BÃI BỎ' in context_str}")
    print(f"[CHECK] Context has sibling points: {'Các điểm khác cùng khoản' in context_str}")
    
    # Verify NĐ 168 appears before NĐ 100 in top results
    nd168_positions = [i for i, r in enumerate(final_results) if "168/2024" in r.uid]
    nd100_positions = [i for i, r in enumerate(final_results) if "100/2019" in r.uid]
    if nd168_positions and nd100_positions:
        print(f"[CHECK] NĐ 168 positions: {nd168_positions}, NĐ 100 positions: {nd100_positions}")
        print(f"[CHECK] NĐ 168 ranked higher than NĐ 100: {min(nd168_positions) < min(nd100_positions)}")
    
    # Generate answer
    answer = generator.generate_rag_answer(query, context_str)
    print(f"\n[ANSWER]:\n{answer[:500]}")
    
    # Check if answer cites NĐ 168
    cites_168 = "168/2024" in answer or "168" in answer
    cites_100_only = "100/2019" in answer and "168" not in answer
    print(f"\n[CHECK] Answer cites NĐ 168: {cites_168}")
    print(f"[CHECK] Answer ONLY cites NĐ 100 (BAD): {cites_100_only}")

embedder.close()
print(f"\n{'='*80}")
print("ALL TESTS COMPLETE")
print(f"{'='*80}")
