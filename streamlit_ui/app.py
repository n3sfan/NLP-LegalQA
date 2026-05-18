"""Streamlit chat UI for the Legal QA API.

Provides a premium dark-themed chat interface that communicates with
the FastAPI backend at ``http://localhost:8000``.

Usage::

    cd streamlit_ui
    pip install -r requirements.txt
    streamlit run app.py
"""

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Hỏi đáp và tư vấn pháp luật về giao thông",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ---- Import Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ---- Global ---- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Header ---- */
.header-container {
    background: linear-gradient(135deg, #1a1f3a 0%, #0d1117 50%, #162447 100%);
    border: 1px solid rgba(79, 139, 249, 0.2);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.header-container::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #4F8BF9, #7C3AED, #4F8BF9);
    background-size: 200% auto;
    animation: shimmer 3s linear infinite;
}
@keyframes shimmer {
    0% { background-position: 200% center; }
    100% { background-position: -200% center; }
}
.header-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #FAFAFA;
    margin: 0;
    letter-spacing: -0.02em;
}
.header-subtitle {
    font-size: 0.9rem;
    color: rgba(250, 250, 250, 0.55);
    margin-top: 0.25rem;
}

/* ---- Chat messages ---- */
.stChatMessage {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    margin-bottom: 0.75rem !important;
}

/* ---- Source cards ---- */
.source-card {
    background: rgba(79, 139, 249, 0.08);
    border: 1px solid rgba(79, 139, 249, 0.2);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s ease;
}
.source-card:hover {
    border-color: rgba(79, 139, 249, 0.5);
}
.source-uid {
    font-weight: 600;
    color: #4F8BF9;
    font-size: 0.85rem;
}
.source-label {
    display: inline-block;
    background: rgba(79, 139, 249, 0.15);
    color: #7AAFFF;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 500;
    margin-left: 0.5rem;
}
.source-score {
    color: rgba(250,250,250,0.5);
    font-size: 0.78rem;
}
.source-snippet {
    color: rgba(250,250,250,0.7);
    font-size: 0.8rem;
    margin-top: 0.4rem;
    line-height: 1.5;
}

/* ---- Timing bar ---- */
.timing-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
}
.timing-chip {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 0.25rem 0.6rem;
    font-size: 0.72rem;
    color: rgba(250,250,250,0.6);
}

/* ---- Sidebar ---- */
.sidebar-section-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: rgba(250,250,250,0.5);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.5rem;
    margin-top: 1rem;
}

/* ---- Status indicator ---- */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 0.4rem;
    vertical-align: middle;
}
.status-ok { background: #22C55E; box-shadow: 0 0 6px rgba(34,197,94,0.5); }
.status-err { background: #EF4444; box-shadow: 0 0 6px rgba(239,68,68,0.5); }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# Sidebar — pipeline controls + status
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Hỏi đáp Pháp luật")
    st.caption("Hỏi đáp và tư vấn luật Giao thông")

    st.markdown('<div class="sidebar-section-title">API Status</div>', unsafe_allow_html=True)
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        status = health.get("status", "unknown")
        provider = health.get("provider", "N/A")
        model = health.get("model", "N/A")
        if status == "ok":
            st.markdown(f'<span class="status-dot status-ok"></span> Connected', unsafe_allow_html=True)
            st.caption(f"Provider: `{provider}`  \nModel: `{model}`")
        else:
            st.markdown(f'<span class="status-dot status-err"></span> {status}', unsafe_allow_html=True)
    except Exception:
        st.markdown('<span class="status-dot status-err"></span> Disconnected', unsafe_allow_html=True)
        st.caption(f"Cannot reach `{API_BASE}`")

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">Pipeline Configuration</div>', unsafe_allow_html=True)

    decompose = st.toggle("Query Decomposition", value=True, help="Split the question into multiple sub-queries")
    hybrid = st.toggle("Hybrid Search", value=True, help="Vector + BM25 keyword search")
    expand = st.toggle("Context Expansion", value=True, help="Expand child content (Clause/Point) for Articles")

    aggregate = st.selectbox("Aggregation", ["rrf", "borda", "max"], index=0, help="Result aggregation strategy")

    labels = st.multiselect(
        "Node Labels",
        ["Article", "Clause", "Point"],
        default=["Article", "Clause", "Point"],
        help="Node types to search",
    )

    st.markdown('<div class="sidebar-section-title">Retrieval Tuning</div>', unsafe_allow_html=True)

    fetch_k = st.slider("Fetch K (candidates per search)", 5, 100, 30, step=5)
    rerank_top = st.slider("Rerank Pool", 5, 50, 15, step=5, help="Number of candidates to rerank via cross-encoder")
    top_k = st.slider("Top K (final results)", 1, 20, 8, help="Number of final results used for answer generation")
    max_history = st.slider("Max History (turns)", 1, 30, 10, help="Maximum conversation turns to keep")

    st.markdown("---")
    if st.button("✕  Xóa cuộc trò chuyện", use_container_width=True, type="secondary"):
        st.session_state.messages.clear()
        st.session_state.chat_history.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="header-container">
    <p class="header-title">Hỏi đáp và tư vấn về luật Giao thông</p>
    <p class="header-subtitle">Nhập môn ngôn ngữ học thống kê và ứng dụng</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Display chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources if available
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])} results)", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="source-uid">{src['uid_formatted']}</span>
                        <span class="source-label">{src['label']}</span>
                        <span class="source-score"> — score: {src['score']}</span>
                        <div class="source-snippet">{src['context_snippet'][:200]}{'…' if len(src['context_snippet']) > 200 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Show timings if available
        if msg["role"] == "assistant" and msg.get("timings"):
            timings = msg["timings"]
            chips = "".join([f'<span class="timing-chip">{k}: {v:.2f}s</span>' for k, v in timings.items()])
            total = sum(timings.values())
            chips += f'<span class="timing-chip" style="border-color:rgba(79,139,249,0.3);color:#7AAFFF;">total: {total:.2f}s</span>'
            st.markdown(f'<div class="timing-bar">{chips}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Đặt câu hỏi..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Analyzing and searching…"):
            try:
                payload = {
                    "query": prompt,
                    "chat_history": st.session_state.chat_history,
                    "decompose": decompose,
                    "hybrid": hybrid,
                    "aggregate": aggregate,
                    "fetch_k": fetch_k,
                    "rerank_top": rerank_top,
                    "top_k": top_k,
                    "max_history": max_history,
                    "labels": labels,
                    "expand": expand,
                }
                resp = requests.post(f"{API_BASE}/chat", json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()

                answer = data.get("answer", "No answer received.")
                sources = data.get("sources", [])
                timings = data.get("timings", {})
                rewritten = data.get("rewritten_query")
                sub_queries = data.get("sub_queries", [])

                # Show rewritten query if different
                if rewritten:
                    st.caption(f"Rewritten query: _{rewritten}_")

                # Show sub-queries
                if sub_queries and len(sub_queries) > 1:
                    with st.expander("Sub-queries", expanded=False):
                        for i, sq in enumerate(sub_queries, 1):
                            st.text(f"  {i}. {sq}")

                # Show answer
                st.markdown(answer)

                # Show sources
                if sources:
                    with st.expander(f"Nguồn tham khảo ({len(sources)} kết quả)", expanded=False):
                        for src in sources:
                            st.markdown(f"""
                            <div class="source-card">
                                <span class="source-uid">{src['uid_formatted']}</span>
                                <span class="source-label">{src['label']}</span>
                                <span class="source-score"> — score: {src['score']}</span>
                                <div class="source-snippet">{src['context_snippet'][:200]}{'…' if len(src['context_snippet']) > 200 else ''}</div>
                            </div>
                            """, unsafe_allow_html=True)

                # Show timings
                if timings:
                    chips = "".join([f'<span class="timing-chip">{k}: {v:.2f}s</span>' for k, v in timings.items()])
                    total = sum(timings.values())
                    chips += f'<span class="timing-chip" style="border-color:rgba(79,139,249,0.3);color:#7AAFFF;">total: {total:.2f}s</span>'
                    st.markdown(f'<div class="timing-bar">{chips}</div>', unsafe_allow_html=True)

                # Save to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "timings": timings,
                })

                # Update chat_history for next request
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

            except requests.exceptions.ConnectionError:
                error_msg = f"Cannot connect to the API server at `{API_BASE}`. Please make sure the server is running."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except requests.exceptions.Timeout:
                error_msg = "Request timeout — the server took too long to respond."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
