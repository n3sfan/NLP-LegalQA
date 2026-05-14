"""FastAPI backend exposing the Legal QA chat pipeline.

Mirrors the CLI ``chat`` command:  rewrite → route → decompose → search →
rerank → graph-boost → expand → generate.

All heavyweight components (Neo4jEmbedder, QueryRewriter, QueryRouter,
AnswerGenerator, VietnameseReranker) are initialised **once** at startup and
shared across requests.  Conversation history is managed client-side — the
caller sends the full ``chat_history`` list with every request.

Usage::

    uv run uvicorn legal_scraper.api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    """Request body for ``POST /chat``."""
    query: str
    chat_history: list[ChatMessage] = []

    # Pipeline flags — mirrors every CLI ``chat`` flag
    decompose: bool = True
    hybrid: bool = True
    aggregate: str = Field(default="rrf", pattern="^(rrf|borda|max)$")
    fetch_k: int = Field(default=30, ge=1, le=200)
    rerank_top: int = Field(default=15, ge=1, le=100)
    top_k: int = Field(default=8, ge=1, le=50)
    max_history: int = Field(default=10, ge=1, le=50)
    labels: list[str] = Field(default=["Article", "Clause", "Point"])
    expand: bool = False
    provider: str | None = None  # "local" | "openrouter" | None (use env)


class SourceItem(BaseModel):
    uid: str
    label: str
    score: float
    uid_formatted: str
    context_snippet: str


class ChatResponse(BaseModel):
    answer: str
    intent: str
    sources: list[SourceItem] = []
    timings: dict[str, float] = {}
    rewritten_query: str | None = None
    sub_queries: list[str] = []


# ---------------------------------------------------------------------------
# Global component references (populated during lifespan)
# ---------------------------------------------------------------------------

_components: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy components once at startup, tear down on shutdown."""
    from legal_scraper.embedder import Neo4jEmbedder
    from legal_scraper.generator import AnswerGenerator
    from legal_scraper.query_rewriter import QueryRewriter
    from legal_scraper.reranker import VietnameseReranker
    from legal_scraper.router import QueryRouter

    print("[api] Initializing components …")
    t0 = time.time()

    embedder = Neo4jEmbedder(
        uri=os.getenv("NEO4J_URI", ""),
        user=os.getenv("NEO4J_USER", ""),
        password=os.getenv("NEO4J_PASSWORD", ""),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )
    rewriter = QueryRewriter()
    router = QueryRouter()
    generator = AnswerGenerator()
    reranker = VietnameseReranker()

    _components["embedder"] = embedder
    _components["rewriter"] = rewriter
    _components["router"] = router
    _components["generator"] = generator
    _components["reranker"] = reranker

    provider = os.getenv("LLM_PROVIDER", "local")
    _components["provider"] = provider
    _components["model_name"] = getattr(rewriter.llm, "model_name", "unknown")
    _components["base_url"] = getattr(rewriter.llm, "openai_api_base", "unknown")

    print(f"[api] Components ready ({time.time() - t0:.1f}s)")
    print(f"[api] LLM Provider: {provider}")
    print(f"[api] Model:    {_components['model_name']}")
    print(f"[api] Base URL: {_components['base_url']}")

    yield  # app is running

    # Shutdown
    embedder.close()
    _components.clear()
    print("[api] Shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Legal QA API",
    description="Vietnamese traffic-law RAG chatbot API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check — returns component status and LLM info."""
    ready = bool(_components)
    return {
        "status": "ok" if ready else "initializing",
        "components_loaded": list(_components.keys()),
        "provider": _components.get("provider"),
        "model": _components.get("model_name"),
        "base_url": _components.get("base_url"),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Full chat pipeline: rewrite → route → decompose → search → rerank → generate."""
    from legal_scraper.embedder import Neo4jEmbedder
    from legal_scraper.retrieval import aggregate_search_results, fetch_context_for_results

    embedder: Neo4jEmbedder = _components["embedder"]
    rewriter = _components["rewriter"]
    router = _components["router"]
    generator = _components["generator"]
    reranker = _components["reranker"]

    timings: dict[str, float] = {}
    sources: list[SourceItem] = []
    sub_query_texts: list[str] = []

    # Trim history to max_history turns (each turn = 2 messages)
    history_dicts = [m.model_dump() for m in req.chat_history]
    max_msgs = req.max_history * 2
    if len(history_dicts) > max_msgs:
        history_dicts = history_dicts[-max_msgs:]

    # --- Step 0: Route the ORIGINAL query first ---
    # This must happen BEFORE rewriting, because the rewriter will
    # incorporate chat history and transform greetings like "hello" into
    # legal questions, preventing the router from recognising them.
    t0 = time.time()
    intent = router.route(req.query)
    timings["route"] = round(time.time() - t0, 3)

    if intent == "reject":
        answer = "Xin lỗi, tôi là chatbot pháp luật giao thông đường bộ Việt Nam. Câu hỏi của bạn nằm ngoài phạm vi tư vấn của tôi."
        return ChatResponse(
            answer=answer,
            intent=intent,
            timings=timings,
        )

    if intent == "direct_answer":
        t1 = time.time()
        answer = generator.generate_direct_answer(req.query)
        timings["generation"] = round(time.time() - t1, 3)
        return ChatResponse(
            answer=answer,
            intent=intent,
            timings=timings,
        )

    # --- Step 1: Rewrite (only for "retrieve" intent) ---
    t1 = time.time()
    rewritten_query = rewriter.rewrite(history_dicts, req.query)
    timings["rewrite"] = round(time.time() - t1, 3)

    # --- intent == "retrieve" ---
    # Step 2: Decompose → Search
    if req.decompose:
        from legal_scraper.query_parser import QueryDecomposer

        t2 = time.time()
        decomposer = QueryDecomposer()
        try:
            sub_queries = decomposer.decompose(rewritten_query)
            sub_queries.append({"query": rewritten_query})
        except Exception:
            sub_queries = [{"query": rewritten_query}]
        timings["decompose"] = round(time.time() - t2, 3)
        sub_query_texts = [sq["query"] for sq in sub_queries]

        raw_results = embedder.multi_search(sub_queries, k=req.fetch_k, hybrid=req.hybrid)
        search_results = aggregate_search_results(raw_results, strategy=req.aggregate)[:req.fetch_k]
        rerank_query = " ".join([sq["query"] for sq in sub_queries[:-1]])
    else:
        t2 = time.time()
        search_fn = embedder.hybrid_search if req.hybrid else embedder.search
        search_results = search_fn(req.labels, rewritten_query, k=req.fetch_k)[:req.fetch_k]
        rerank_query = rewritten_query
        timings["search"] = round(time.time() - t2, 3)

    if not search_results:
        return ChatResponse(
            answer="Không tìm thấy kết quả phù hợp trong cơ sở dữ liệu pháp luật.",
            intent=intent,
            timings=timings,
            rewritten_query=rewritten_query if rewritten_query != req.query else None,
            sub_queries=sub_query_texts,
        )

    # Step 3: Fetch context & rerank
    t3 = time.time()
    rerank_pool = min(req.rerank_top, len(search_results))
    context_map = fetch_context_for_results(embedder, search_results[:rerank_pool], include_hierarchy=True)
    documents = [context_map.get((r.uid, r.label), "") for r in search_results[:rerank_pool]]
    reranked_indices = reranker.rerank(rerank_query, documents, top_k=rerank_pool)
    timings["rerank"] = round(time.time() - t3, 3)

    # Step 3b: Graph-based score adjustments
    t3b = time.time()
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
                recency_bonus = max(0, 2.0 - 0.3 * years_old)
                score += recency_bonus
            except ValueError:
                pass

        adjusted_indices.append((idx, score))

    adjusted_indices.sort(key=lambda x: x[1], reverse=True)
    final_results = [search_results[idx] for idx, _ in adjusted_indices[:req.top_k]]
    final_scores = adjusted_indices[:req.top_k]
    timings["graph_boost"] = round(time.time() - t3b, 3)

    # Step 4: Build context & generate
    top_k_uids = [r.uid for r in final_results]
    amends_map = embedder.fetch_amends(top_k_uids)

    # Context expansion (controllable via expand flag)
    siblings_map: dict = {}
    children_map: dict = {}
    if req.expand:
        point_uids = [r.uid for r in final_results if r.label == "Point"]
        siblings_map = embedder.fetch_sibling_points(point_uids) if point_uids else {}
        parent_uids = [r.uid for r in final_results if r.label in ("Article", "Clause")]
        children_map = embedder.fetch_children_context(parent_uids) if parent_uids else {}

    # Build source items for the response
    for r, (_, adj_score) in zip(final_results, final_scores):
        ctx = context_map.get((r.uid, r.label), "")
        sources.append(SourceItem(
            uid=r.uid,
            label=r.label,
            score=round(adj_score, 4),
            uid_formatted=Neo4jEmbedder.format_uid_vn(r.uid),
            context_snippet=ctx[:300] if ctx else "",
        ))

    # Build full context string for generation
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

        if r.uid in children_map:
            ctx += f"\n\n[Nội dung chi tiết]:\n{children_map[r.uid]}"

        amends = amends_map.get(r.uid, [])
        if amends:
            amend_str = "\n".join([f"Đã được sửa đổi/bổ sung: {a['amending_content']}" for a in amends])
            ctx = f"{ctx}\n\n[LƯU Ý - NỘI DUNG SỬA ĐỔI]:\n{amend_str}"
        context_blocks.append(ctx)

    context_str = "\n\n---\n\n".join(context_blocks)

    t4 = time.time()
    answer = generator.generate_rag_answer(req.query, context_str)
    timings["generation"] = round(time.time() - t4, 3)

    # Sanitize surrogates
    answer = answer.encode("utf-8", errors="replace").decode("utf-8")

    return ChatResponse(
        answer=answer,
        intent=intent,
        sources=sources,
        timings=timings,
        rewritten_query=rewritten_query if rewritten_query != req.query else None,
        sub_queries=sub_query_texts,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the API server via ``uv run python -m legal_scraper.api``."""
    import uvicorn
    uvicorn.run(
        "legal_scraper.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
