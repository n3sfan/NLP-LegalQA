"""Integration tests for decompose → multi_search → aggregate → fetch context."""

from dotenv import load_dotenv
import os
import sys
import pytest

sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from legal_scraper.embedder import Neo4jEmbedder, SearchResult
from legal_scraper.reranker import VietnameseReranker
from legal_scraper import retrieval

# Load test environment
load_dotenv()


@pytest.fixture(scope="module")
def embedder():
    e = Neo4jEmbedder(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"],
        database=os.environ.get("NEO4J_DATABASE", "neo4j"),
    )
    yield e
    e.close()


def test_aggregate_search_results_rrf(embedder):
    """Test RRF aggregation with overlapping results."""
    raw = {
        0: [
            SearchResult(uid="doc1::article::1", label="Article", score=0.9),
            SearchResult(uid="doc2::article::2", label="Article", score=0.8),
        ],
        1: [
            SearchResult(uid="doc1::article::1", label="Article", score=0.85),
            SearchResult(uid="doc3::clause::1", label="Clause", score=0.7),
        ],
    }
    agg = retrieval.aggregate_search_results(raw, strategy="rrf")
    assert len(agg) == 3
    assert agg[0].uid == "doc1::article::1"
    scores = [r.score for r in agg]
    assert scores == sorted(scores, reverse=True)


def test_aggregate_search_results_max(embedder):
    """Test max-score aggregation."""
    raw = {
        0: [SearchResult(uid="doc1", label="Article", score=0.5)],
        1: [SearchResult(uid="doc1", label="Article", score=0.9)],
    }
    agg = retrieval.aggregate_search_results(raw, strategy="max")
    assert len(agg) == 1
    assert agg[0].score == 0.9


def test_aggregate_search_results_borda(embedder):
    """Test Borda (average) aggregation."""
    raw = {
        0: [SearchResult(uid="doc1", label="Article", score=0.5)],
        1: [SearchResult(uid="doc1", label="Article", score=0.9)],
    }
    agg = retrieval.aggregate_search_results(raw, strategy="borda")
    assert len(agg) == 1
    assert agg[0].score == pytest.approx((0.5 + 0.9) / 2)


def test_aggregate_deduplication(embedder):
    """Test that same document appearing in multiple sub-queries is deduplicated."""
    raw = {
        0: [SearchResult(uid="same::doc", label="Clause", score=0.9)],
        1: [SearchResult(uid="same::doc", label="Clause", score=0.85)],
        2: [SearchResult(uid="same::doc", label="Clause", score=0.7)],
    }
    agg = retrieval.aggregate_search_results(raw, strategy="max")
    assert len(agg) == 1
    assert agg[0].uid == "same::doc"


def test_fetch_context_for_results_hierarchy(embedder):
    """Test context fetching with hierarchy."""
    results = [SearchResult(uid="56/2024/QH15::article::1", label="Article", score=0.9)]
    ctx = retrieval.fetch_context_for_results(embedder, results, include_hierarchy=True)
    key = ("56/2024/QH15::article::1", "Article")
    assert key in ctx
    content = ctx[key]
    assert len(content) > 0
    assert any(h in content for h in ["Điều", "Chương", "Phần"])


def test_fetch_context_for_results_no_hierarchy(embedder):
    """Test context fetching without hierarchy."""
    results = [SearchResult(uid="56/2024/QH15::article::1", label="Article", score=0.9)]
    ctx = retrieval.fetch_context_for_results(embedder, results, include_hierarchy=False)
    key = ("56/2024/QH15::article::1", "Article")
    assert key in ctx
    assert len(ctx[key]) > 0


def test_retrieve_and_build_context_no_decompose(embedder):
    """Test shared pipeline with decomposition disabled."""
    from legal_scraper.retrieval import retrieve_and_build_context, RetrievalResult
    from legal_scraper.reranker import VietnameseReranker

    reranker = VietnameseReranker(device="cpu")
    query = "Vượt đèn đỏ bị xử phạt thế nào"
    rr = retrieve_and_build_context(
        embedder=embedder,
        reranker=reranker,
        query=query,
        decompose=False,
        top_k=3,
    )
    assert isinstance(rr, RetrievalResult)
    assert len(rr.final_results) <= 3
    for r in rr.final_results:
        assert isinstance(r, SearchResult)
        assert r.uid
        assert r.label in ["Article", "Clause", "Point"]
    assert len(rr.context_str) > 0


def test_retrieve_and_build_context_with_decompose(embedder):
    """Test shared pipeline with decomposition enabled."""
    from legal_scraper.retrieval import retrieve_and_build_context, RetrievalResult
    from legal_scraper.reranker import VietnameseReranker

    reranker = VietnameseReranker(device="cpu")
    query = "Không đội mũ bảo hiểm và vượt đèn đỏ bị xử phạt thế nào"
    rr = retrieve_and_build_context(
        embedder=embedder,
        reranker=reranker,
        query=query,
        decompose=True,
        top_k=5,
    )
    assert isinstance(rr, RetrievalResult)
    assert len(rr.final_results) <= 5
    assert len(rr.sub_queries) > 0
    assert len(rr.context_str) > 0


def test_retrieve_and_build_context_heuristic_rerank(embedder):
    """Test that heuristic re-ranking timings are recorded."""
    from legal_scraper.retrieval import retrieve_and_build_context, RetrievalResult
    from legal_scraper.reranker import VietnameseReranker

    reranker = VietnameseReranker(device="cpu")
    query = "Phạt xe quá tải"
    rr = retrieve_and_build_context(
        embedder=embedder,
        reranker=reranker,
        query=query,
        decompose=False,
        top_k=3,
    )
    assert isinstance(rr, RetrievalResult)
    assert "heuristic_rerank" in rr.timings
    assert "rerank" in rr.timings


def test_decompose_demo(embedder, capsys):
    """Demo: full decompose → multi_search → fetch → rerank workflow."""
    from legal_scraper.query_parser import QueryDecomposer
    decomposer = QueryDecomposer()
    
    query = "Tui nhậu xỉn xong lái xe mà đụng xe, làm hư điện thoại của người khác xong làm người ta bị thương, đồng thời vượt đèn đỏ thì bị phạt bao nhiêu?"
    try:
        sub_queries = decomposer.decompose(query)
        success = True
    except Exception as e:
        print(f"Decomposition failed: {e}")
        sub_queries = []
        success = False

    print("\n=== SUB-QUERIES ===")
    for i, sq in enumerate(sub_queries):
        print(f"  [{i}] {sq.get('query', sq)}")
    print(f"\nSuccess: {success}")

    if not sub_queries:
        pytest.skip("Decomposition failed, skipping rerank demo")
        return

    results = embedder.multi_search(sub_queries, k=5)
    all_uids = [r.uid for hits in results.values() for r in hits]
    all_labels = list({r.label for hits in results.values() for r in hits})
    node_contents = embedder.fetch_nodes(all_uids, list(set(all_labels)))

    reranker = VietnameseReranker(device="cpu")

    print("\n=== SCORE COMPARISON ===")
    for idx, hits in results.items():
        sq = sub_queries[idx]
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])