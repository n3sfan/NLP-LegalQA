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
        local_model_url=os.environ.get("LOCAL_MODEL_URL", "https://vitalize-compacter-nephew.ngrok-free.dev/generate"),
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


def test_full_retrieval_pipeline_no_decompose(embedder):
    """Test pipeline with decomposition disabled."""
    query = "Vượt đèn đỏ phạt bao nhiêu?"
    results = retrieval.full_retrieval_pipeline(
        embedder=embedder,
        query=query,
        k=3,
        decomposition=False,
        aggregation_strategy="rrf",
        rerank_top=0,
    )
    assert isinstance(results, list)
    assert len(results) <= 3
    for r in results:
        assert isinstance(r, SearchResult)
        assert r.uid
        assert r.label in ["Article", "Clause", "Point"]
        assert r.score >= 0


def test_full_retrieval_pipeline_with_decompose(embedder):
    """Test pipeline with decomposition enabled."""
    query = "Không đội mũ bảo hiểm và vượt đèn đỏ thì bị phạt thế nào?"
    results = retrieval.full_retrieval_pipeline(
        embedder=embedder,
        query=query,
        k=5,
        decomposition=True,
        aggregation_strategy="rrf",
        rerank_top=0,
    )
    assert isinstance(results, list)
    assert len(results) <= 5
    for r in results:
        assert isinstance(r, SearchResult)
        assert r.uid
        assert r.score >= 0


def test_full_retrieval_pipeline_with_rerank(embedder):
    """Test pipeline with reranking enabled."""
    query = "Phạt xe quá tải là bao nhiêu?"
    results = retrieval.full_retrieval_pipeline(
        embedder=embedder,
        query=query,
        k=3,
        decomposition=False,
        aggregation_strategy="rrf",
        rerank_top=5,
    )
    assert isinstance(results, list)
    assert len(results) <= 3
    for r in results:
        assert isinstance(r, SearchResult)
        assert r.score >= 0


def test_decompose_demo(embedder, capsys):
    """Demo: full decompose → multi_search → fetch → rerank workflow."""
    query = "Tui nhậu xỉn xong lái xe mà đụng xe, làm hư điện thoại của người khác xong làm người ta bị thương, đồng thời vượt đèn đỏ thì bị phạt bao nhiêu?"
    decomp = embedder.decompose_query_debug(query)

    print("=== RAW LLM RESPONSE ===")
    print(decomp.reasoning)
    print("\n=== SUB-QUERIES ===")
    for i, sq in enumerate(decomp.sub_queries):
        print(f"  [{i}] {sq.get('query', sq)}")
    print(f"\nSuccess: {decomp.success}")

    if not decomp.sub_queries:
        pytest.skip("Decomposition failed, skipping rerank demo")
        return

    results = embedder.multi_search(decomp.sub_queries, k=5)
    all_uids = [r.uid for hits in results.values() for r in hits]
    all_labels = list({r.label for hits in results.values() for r in hits})
    node_contents = embedder.fetch_nodes(all_uids, list(set(all_labels)))

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])