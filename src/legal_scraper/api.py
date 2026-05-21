"""FastAPI backend exposing the Legal QA chat pipeline.

Mirrors the CLI ``chat`` command:  rewrite → route → decompose → search →
rerank → heuristic-rerank → expand → generate.

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
from fastapi import FastAPI
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
    cypher_query: str | None = None


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
    from legal_scraper.text2cypher import Neo4jGeminiQuery

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

    try:
        cypher_tool = Neo4jGeminiQuery(
            url=os.getenv("NEO4J_URI", ""),
            user=os.getenv("NEO4J_USER", ""),
            password=os.getenv("NEO4J_PASSWORD", ""),
            gemini_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        )
    except Exception as e:
        print(f"[api] Warning: cypher_tool init failed: {e}")
        cypher_tool = None

    _components["embedder"] = embedder
    _components["rewriter"] = rewriter
    _components["router"] = router
    _components["generator"] = generator
    _components["reranker"] = reranker
    _components["cypher_tool"] = cypher_tool

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

    # --- Step 1: Rewrite (only for "retrieve" and "cypher_query" intent) ---
    t1 = time.time()
    rewritten_query = rewriter.rewrite(history_dicts, req.query)
    timings["rewrite"] = round(time.time() - t1, 3)

    if intent == "cypher_query":
        t2 = time.time()
        cypher_tool = _components.get("cypher_tool")
        if not cypher_tool:
            return ChatResponse(
                answer="Xin lỗi, tính năng tra cứu bằng Cypher hiện không khả dụng (chưa được cấu hình).",
                intent=intent,
                timings=timings,
            )
        
        raw_result, generated_cypher = cypher_tool.run(rewritten_query)
        if isinstance(raw_result, str):
            answer = raw_result
        else:
            answer = generator.generate_cypher_answer(rewritten_query, str(raw_result))
            
        timings["cypher"] = round(time.time() - t2, 3)
        return ChatResponse(
            answer=answer,
            intent=intent,
            timings=timings,
            rewritten_query=rewritten_query if rewritten_query != req.query else None,
            cypher_query=generated_cypher,
        )

    # --- intent == "retrieve" ---
    from legal_scraper.retrieval import retrieve_and_build_context, RetrievalResult

    retrieval_result = retrieve_and_build_context(
        embedder=embedder,
        reranker=reranker,
        query=rewritten_query,
        decompose=req.decompose,
        hybrid=req.hybrid,
        aggregate=req.aggregate,
        fetch_k=req.fetch_k,
        rerank_top=req.rerank_top,
        top_k=req.top_k,
        labels=req.labels,
        expand=req.expand,
    )
    timings.update(retrieval_result.timings)
    sub_query_texts = retrieval_result.sub_queries

    if not retrieval_result.final_results:
        return ChatResponse(
            answer="Không tìm thấy kết quả phù hợp trong cơ sở dữ liệu pháp luật.",
            intent=intent,
            timings=timings,
            rewritten_query=rewritten_query if rewritten_query != req.query else None,
            sub_queries=sub_query_texts,
        )

    # Build source items for the response
    for r, (_, adj_score) in zip(retrieval_result.final_results, retrieval_result.final_scores):
        ctx = retrieval_result.context_map.get((r.uid, r.label), "")
        sources.append(SourceItem(
            uid=r.uid,
            label=r.label,
            score=round(adj_score, 4),
            uid_formatted=Neo4jEmbedder.format_uid_vn(r.uid),
            context_snippet=ctx[:300] if ctx else "",
        ))

    # Generate answer
    t4 = time.time()
    answer = generator.generate_rag_answer(req.query, retrieval_result.context_str, rewritten_query=rewritten_query)
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
