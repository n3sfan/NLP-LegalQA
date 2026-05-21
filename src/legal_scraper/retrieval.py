"""Retrieval pipeline orchestration: aggregation, context fetching, full pipeline.

This module provides the shared retrieval pipeline used by both the FastAPI
backend (``api.py``) and the CLI (``cli.py``).  All retrieval logic lives here
so that bug fixes and feature additions only need to happen in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from legal_scraper.embedder import Neo4jEmbedder, SearchResult
from legal_scraper.reranker import VietnameseReranker


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Output of the shared retrieval pipeline.

    Contains everything the caller needs to build an API response or CLI
    output: final ranked results, context string for the LLM, timing info,
    sub-query details, and per-result metadata.
    """

    # Core outputs
    final_results: list[SearchResult] = field(default_factory=list)
    final_scores: list[tuple[int, float]] = field(default_factory=list)
    context_str: str = ""

    # Metadata for the caller
    sub_queries: list[str] = field(default_factory=list)
    rerank_query: str = ""

    # Per-result maps (exposed for CLI debug output / API source items)
    context_map: dict[tuple[str, str], str] = field(default_factory=dict)
    abolished_map: dict[str, list[str]] = field(default_factory=dict)
    amends_map: dict[str, list[dict]] = field(default_factory=dict)
    siblings_map: dict[str, str] = field(default_factory=dict)
    children_map: dict[str, str] = field(default_factory=dict)

    # Timing breakdown (seconds)
    timings: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Low-level helpers (unchanged, used by eval scripts too)
# ---------------------------------------------------------------------------

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


def build_context_str(
    final_results: list[SearchResult],
    context_map: dict[tuple[str, str], str],
    abolished_map: dict[str, list[str]] | None = None,
    amends_map: dict[str, list[dict]] | None = None,
    siblings_map: dict[str, str] | None = None,
    children_map: dict[str, str] | None = None,
) -> str:
    """Build the final LLM context string from selected retrieval results."""
    abolished_map = abolished_map or {}
    amends_map = amends_map or {}
    siblings_map = siblings_map or {}
    children_map = children_map or {}

    context_blocks: list[str] = []
    for r in final_results:
        ctx = context_map.get((r.uid, r.label), "")

        # Tag abolished provisions
        abolished_types = abolished_map.get(r.uid, [])
        if "bãi bỏ" in abolished_types:
            ctx = f"[ĐÃ BỊ BÃI BỎ bởi văn bản mới hơn]\n{ctx}"
        elif "thay thế" in abolished_types:
            ctx = f"[ĐÃ BỊ THAY THẾ bởi văn bản mới hơn]\n{ctx}"

        # Sibling points
        if r.uid in siblings_map:
            ctx += f"\n\n[Các điểm khác cùng khoản]:\n{siblings_map[r.uid]}"

        # Children content
        if r.uid in children_map:
            ctx += f"\n\n[Nội dung chi tiết]:\n{children_map[r.uid]}"

        # Amends
        amends = amends_map.get(r.uid, [])
        if amends:
            amend_str = "\n".join(
                [f"Đã được sửa đổi/bổ sung bởi {Neo4jEmbedder.format_uid_vn(a['amending_uid'])}: {a['amending_content']}" for a in amends]
            )
            ctx = f"{ctx}\n\n[LƯU Ý - NỘI DUNG SỬA ĐỔI]:\n{amend_str}"

        context_blocks.append(ctx)

    return "\n\n---\n\n".join(context_blocks)


def _label_from_uid(uid: str) -> str:
    if "::point::" in uid:
        return "Point"
    if "::clause::" in uid:
        return "Clause"
    return "Article"


def build_context_str_for_uids(
    embedder: Neo4jEmbedder,
    uids: list[str],
    expand: bool = False,
) -> str:
    """Build retrieval-style context_str when the caller already has final UIDs."""
    final_results = [
        SearchResult(uid=uid, label=_label_from_uid(uid), score=0.0)
        for uid in uids
    ]
    context_map = fetch_context_for_results(embedder, final_results, include_hierarchy=True)
    abolished_map = embedder.fetch_abolished_uids(uids)
    amends_map = embedder.fetch_amends(uids)

    siblings_map: dict[str, str] = {}
    children_map: dict[str, str] = {}
    if expand:
        point_uids = [r.uid for r in final_results if r.label == "Point"]
        siblings_map = embedder.fetch_sibling_points(point_uids) if point_uids else {}
        parent_uids = [r.uid for r in final_results if r.label in ("Article", "Clause")]
        children_map = embedder.fetch_children_context(parent_uids) if parent_uids else {}

    return build_context_str(
        final_results,
        context_map,
        abolished_map=abolished_map,
        amends_map=amends_map,
        siblings_map=siblings_map,
        children_map=children_map,
    )


# ---------------------------------------------------------------------------
# Shared retrieval pipeline
# ---------------------------------------------------------------------------

def retrieve_and_build_context(
    *,
    embedder: Neo4jEmbedder,
    reranker: VietnameseReranker,
    query: str,
    decompose: bool = True,
    hybrid: bool = True,
    aggregate: str = "rrf",
    fetch_k: int = 30,
    rerank_top: int = 15,
    top_k: int = 8,
    labels: list[str] | None = None,
    expand: bool = False,
    heuristic: bool = True,
) -> RetrievalResult:
    """Execute the full retrieval pipeline and build LLM context.

    Steps:
        1. Decompose → multi-search  (or single search)
        2. Aggregate results (RRF / Borda / Max)
        3. Fetch hierarchy context & cross-encoder rerank
        4. Post-retrieval heuristic re-ranking (abolished penalty + recency bonus)
        5. Context expansion (sibling points, children content)
        6. Build final context string with abolished/amends tags

    This is the single source of truth for the retrieval pipeline.  Both
    ``api.py`` and ``cli.py`` call this function.

    Args:
        embedder: Neo4jEmbedder instance (must be open).
        reranker: VietnameseReranker instance.
        query: The search query (rewritten query for multi-turn, original for
            single-turn).  NOT the original user question — that is passed
            separately to the generator.
        decompose: Whether to decompose the query into sub-queries.
        hybrid: Whether to use hybrid search (vector + BM25).
        aggregate: Aggregation strategy ("rrf", "borda", "max").
        fetch_k: Number of initial candidates per search.
        rerank_top: Number of candidates to rerank via cross-encoder.
        top_k: Final number of results for LLM context.
        labels: Node labels to search (default: Article, Clause, Point).
        expand: Whether to expand context with sibling points and children.
        heuristic: Whether to apply post-retrieval heuristic re-ranking
            (abolished penalty + recency bonus).  Set to False for ablation.

    Returns:
        A ``RetrievalResult`` containing the final results, context string,
        timings, and all intermediate maps.
    """
    import time

    if labels is None:
        labels = ["Article", "Clause", "Point"]

    result = RetrievalResult()
    timings: dict[str, float] = {}

    # ── Step 1: Decompose → Search ──────────────────────────────────────

    if decompose:
        from legal_scraper.query_parser import QueryDecomposer

        t = time.time()
        decomposer = QueryDecomposer()
        try:
            sub_queries = decomposer.decompose(query)
            sub_queries.append({"query": query})
        except Exception:
            sub_queries = [{"query": query}]
        timings["decompose"] = round(time.time() - t, 3)

        result.sub_queries = [sq["query"] for sq in sub_queries]

        raw_results = embedder.multi_search(sub_queries, k=fetch_k, hybrid=hybrid)
        search_results = aggregate_search_results(raw_results, strategy=aggregate)[:fetch_k]
        rerank_query = " ".join([sq["query"] for sq in sub_queries[:-1]])
    else:
        t = time.time()
        search_fn = embedder.hybrid_search if hybrid else embedder.search
        search_results = search_fn(labels, query, k=fetch_k)[:fetch_k]
        rerank_query = query
        timings["search"] = round(time.time() - t, 3)

    result.rerank_query = rerank_query

    if not search_results:
        result.timings = timings
        return result

    # ── Step 2: Fetch context & cross-encoder rerank ────────────────────

    t = time.time()
    rerank_pool = min(rerank_top, len(search_results))
    context_map = fetch_context_for_results(
        embedder, search_results[:rerank_pool], include_hierarchy=True
    )
    documents = [context_map.get((r.uid, r.label), "") for r in search_results[:rerank_pool]]
    reranked_indices = reranker.rerank(rerank_query, documents, top_k=rerank_pool)
    timings["rerank"] = round(time.time() - t, 3)

    # ── Step 3: Post-retrieval heuristic re-ranking ─────────────────────

    if heuristic:
        t = time.time()
        pool_uids = [search_results[idx].uid for idx, _ in reranked_indices]

        abolished_map = embedder.fetch_abolished_uids(pool_uids)
        doc_ids = list({uid.split("::")[0] for uid in pool_uids})
        effect_dates = embedder.fetch_doc_effect_dates(doc_ids)
        today = date.today()

        adjusted_indices: list[tuple[int, float]] = []
        for idx, score in reranked_indices:
            uid = search_results[idx].uid
            doc_id = uid.split("::")[0]

            # Additive penalty for abolished/replaced provisions
            amend_types = abolished_map.get(uid, [])
            if "bãi bỏ" in amend_types:
                score -= 5.0
            elif "thay thế" in amend_types:
                score -= 3.0

            # Additive recency boost: newer documents score higher
            eff_str = effect_dates.get(doc_id)
            if eff_str:
                try:
                    eff_date = datetime.strptime(eff_str, "%Y-%m-%d").date()
                    years_old = max(0, (today - eff_date).days) / 365.0
                    recency_bonus = max(0, 2.0 - 0.3 * years_old)
                    score += recency_bonus
                except ValueError:
                    pass

            adjusted_indices.append((idx, score))

        adjusted_indices.sort(key=lambda x: x[1], reverse=True)
        timings["heuristic_rerank"] = round(time.time() - t, 3)
    else:
        # Skip heuristic scoring: use cross-encoder scores directly
        adjusted_indices = list(reranked_indices)
        # Still fetch abolished_map — needed for context tagging in Step 5
        pool_uids = [search_results[idx].uid for idx, _ in reranked_indices]
        abolished_map = embedder.fetch_abolished_uids(pool_uids)

    final_results = [search_results[idx] for idx, _ in adjusted_indices[:top_k]]
    final_scores = adjusted_indices[:top_k]

    # ── Step 4: Fetch amends & expand context ───────────────────────────

    top_k_uids = [r.uid for r in final_results]
    amends_map = embedder.fetch_amends(top_k_uids)

    siblings_map: dict[str, str] = {}
    children_map: dict[str, str] = {}
    if expand:
        point_uids = [r.uid for r in final_results if r.label == "Point"]
        siblings_map = embedder.fetch_sibling_points(point_uids) if point_uids else {}
        parent_uids = [r.uid for r in final_results if r.label in ("Article", "Clause")]
        children_map = embedder.fetch_children_context(parent_uids) if parent_uids else {}

    # ── Step 5: Build context string for LLM ────────────────────────────

    context_str = build_context_str(
        final_results,
        context_map,
        abolished_map=abolished_map,
        amends_map=amends_map,
        siblings_map=siblings_map,
        children_map=children_map,
    )

    # ── Pack result ─────────────────────────────────────────────────────

    result.final_results = final_results
    result.final_scores = final_scores
    result.context_str = context_str
    result.context_map = context_map
    result.abolished_map = abolished_map
    result.amends_map = amends_map
    result.siblings_map = siblings_map
    result.children_map = children_map
    result.timings = timings

    return result
