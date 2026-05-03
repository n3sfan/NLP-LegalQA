"""Retrieval pipeline orchestration: aggregation, context fetching, full pipeline."""

from typing import Dict, List, Optional, Tuple
from legal_scraper.embedder import Neo4jEmbedder, SearchResult
from legal_scraper.reranker import VietnameseReranker


def aggregate_search_results(
    results_dict: Dict[int, List[SearchResult]],
    strategy: str = "rrf",
    top_k: Optional[int] = None,
) -> List[SearchResult]:
    """Deduplicate and fuse scores from multiple sub-query results.

    Args:
        results_dict: Dict mapping sub-query index to list of SearchResult.
        strategy: "rrf", "borda", or "max".
        top_k: Optional limit on final results.

    Returns:
        Deduplicated and fused SearchResult list, sorted descending.
    """
    # Collect scores per (uid, label) with ranks
    scores_with_ranks: Dict[Tuple[str, str], List[Tuple[int, int, float]]] = {}
    result_map: Dict[Tuple[str, str], SearchResult] = {}

    for sq_idx, results in results_dict.items():
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        for rank, r in enumerate(sorted_results):
            key = (r.uid, r.label)
            if key not in result_map:
                result_map[key] = r
            if key not in scores_with_ranks:
                scores_with_ranks[key] = []
            scores_with_ranks[key].append((sq_idx, rank + 1, r.score))

    fused_results = []
    for key, entries in scores_with_ranks.items():
        base = result_map[key]
        if strategy == "rrf":
            score = sum(1.0 / (rank + 60) for _, rank, _ in entries)
        elif strategy == "borda":
            score = sum(score for _, _, score in entries) / len(entries)
        elif strategy == "max":
            score = max(score for _, _, score in entries)
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")
        fused_results.append(SearchResult(uid=base.uid, label=base.label, score=score))

    fused_results.sort(key=lambda r: r.score, reverse=True)
    return fused_results[:top_k] if top_k else fused_results


def fetch_context_for_results(
    embedder: Neo4jEmbedder,
    search_results: List[SearchResult],
    include_hierarchy: bool = True,
) -> Dict[Tuple[str, str], str]:
    """Fetch formatted context for a list of SearchResult.

    Args:
        embedder: Neo4jEmbedder instance.
        search_results: List of SearchResult from search/aggregate.
        include_hierarchy: If True, fetch full hierarchy; else just node content.

    Returns:
        Dict keyed by (uid, label) -> formatted context string.
    """
    uids = list({r.uid for r in search_results})
    labels = list({r.label for r in search_results})

    if include_hierarchy:
        hierarchy_map = embedder.fetch_node_hierarchy(uids)
        return {(r.uid, r.label): hierarchy_map.get(r.uid, "") for r in search_results}
    else:
        node_map = embedder.fetch_nodes(uids, labels)
        search_keys = {(r.uid, r.label) for r in search_results}
        result: Dict[Tuple[str, str], str] = {}
        for (uid, label), data in node_map.items():
            if (uid, label) in search_keys:
                title = data.get("title") or ""
                content = data.get("content") or ""
                text = (f"{title}\n{content}" if title else content).strip()
                result[(uid, label)] = text
        return result


def full_retrieval_pipeline(
    embedder: Neo4jEmbedder,
    query: str,
    k: int = 5,
    decomposition: bool = True,
    aggregation_strategy: str = "rrf",
    rerank_top: int = 0,
    rerank_query: Optional[str] = None,
    include_hierarchy: bool = True,
) -> List[SearchResult]:
    """Execute the complete retrieval pipeline.

    Steps: decompose (optional) → parallel search → aggregate → optional rerank.

    Args:
        embedder: Neo4jEmbedder instance.
        query: Original user question.
        k: Final number of results to return.
        decomposition: Whether to decompose complex queries.
        aggregation_strategy: "rrf", "borda", or "max".
        rerank_top: If > 0, rerank top N aggregated results with cross-encoder.
        rerank_query: Query to use for reranking (defaults to original query).
        include_hierarchy: Whether to fetch full hierarchy for context.

    Returns:
        Final list of SearchResult, sorted by final score.
    """
    if decomposition:
        from legal_scraper.query_parser import QueryDecomposer
        try:
            decomposer = QueryDecomposer()
            sub_queries = decomposer.decompose(query)
        except Exception as e:
            print(f"Decomposition failed, falling back to original query: {e}")
            sub_queries = [{"query": query}]
    else:
        sub_queries = [{"query": query}]

    raw_results = embedder.multi_search(sub_queries, k=k * 2)
    aggregated = aggregate_search_results(raw_results, strategy=aggregation_strategy)[: k * 2]

    if rerank_top > 0 and aggregated:
        rerank_q = rerank_query or query
        context_map = fetch_context_for_results(embedder, aggregated[:rerank_top], include_hierarchy=False)
        docs = [context_map.get((r.uid, r.label), "") for r in aggregated[:rerank_top]]
        reranker = VietnameseReranker(device="cpu")
        reranked_indices = reranker.rerank(rerank_q, docs, top_k=len(docs), batch_size=4)
        reranked = [aggregated[idx] for idx, _ in reranked_indices]
        for i, (_, score) in enumerate(reranked_indices):
            reranked[i] = SearchResult(uid=reranked[i].uid, label=reranked[i].label, score=score)
        final = reranked
    else:
        final = aggregated

    return final[:k]
