"""
Voter Evaluation on RAG Results.

Evaluates how well the majority voter classifies retrieved documents
as relevant (Có) or irrelevant (Không) for each question in row_results.csv.

Supports three backends:
    --backend=llama_cpp  (default) — loads GGUF directly via llama-cpp-python
    --backend=ollama     — local Ollama server
    --backend=vllm      — vLLM OpenAI-compatible server (one per voter on 8000+i)

Usage (llama_cpp — default):
    uv run python -m src.llm.eval_voter \
        --input eval_results/row_results.csv \
        --output eval_results/ \
        --uri "neo4j+ssc://host:7687" \
        --user neo4j --password "..." --database neo4j \
        --n-voters 3 \
        --models /path/to/Qwen3-4B-Q4_K_M.gguf /path/to/nanbeige4.1-Q4_K_M.gguf \
        --n-gpu-layers 33 \
        --n-ctx 4096

Usage (ollama):
    uv run python -m src.llm.eval_voter \
        --input eval_results/row_results.csv \
        --output eval_results/ \
        --backend ollama \
        --n-voters 3 \
        --models qwen3:4b nanbeige4b:latest

Usage (vllm):
    # Each voter i uses http://localhost:{8000+i}/v1  (i = 0 .. n-1)
    # Servers must be started separately on those ports beforehand.
    uv run python -m src.llm.eval_voter \
        --input eval_results/row_results.csv \
        --output eval_results/ \
        --backend vllm \
        --n-voters 3 \
        --model Qwen/Qwen3-4B \
        --api-key vllm-secret-key

    # with custom prompt template:
    --prompt-template src/llm/prompt_classify_zero_shot.md
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

# Repo root is the parent of src/ (NLP-LegalQA/)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from voter import (  # noqa: E402  (sys.path set above)
    VoteResult,
    load_prompt_template,
)
from eval_qa_utils import get_payload_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Silence noisy third-party logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Relevance (same logic as eval_rag.py)
# ─────────────────────────────────────────────────────────────────────────────

def is_relevant(retrieved_uid: str, reference: str) -> bool:
    """Return True if `retrieved_uid` shares a prefix with `reference`."""
    return retrieved_uid.startswith(reference)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt template context manager (swap load_prompt_template at runtime)
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _swap_prompt_template(path: str | None):
    """Temporarily override load_prompt_template if a custom path is given."""
    if path is None:
        yield
        return
    custom_text = Path(path).read_text(encoding="utf-8")
    original = load_prompt_template
    import voter as voter_module
    voter_module.load_prompt_template = lambda: custom_text
    try:
        yield
    finally:
        voter_module.load_prompt_template = original


# ─────────────────────────────────────────────────────────────────────────────
# UID helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parent_article_uid(uid: str) -> str | None:
    """Extract parent Article UID from a Clause or Point UID."""
    # Format: {doc_identity}::article::{N}::clause::{M}::point::{letter}
    # We want: {doc_identity}::article::{N}
    parts = uid.split("::")
    if len(parts) >= 3 and parts[1] == "article":
        return "::".join(parts[:3])
    return None


def _parent_clause_uid(uid: str) -> str | None:
    """Extract parent Clause UID from a Point UID."""
    # Format: {doc_identity}::article::{N}::clause::{M}::point::{letter}
    # We want: {doc_identity}::article::{N}::clause::{M}
    parts = uid.split("::")
    if len(parts) >= 5 and parts[1] == "article" and parts[3] == "clause":
        return "::".join(parts[:5])
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Neo4j helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_nodes_sync(driver, uids: list[str]) -> dict[str, dict]:
    """Fetch uid/title/content/label/doc_identity for all UIDs in one Cypher call."""
    if not uids:
        return {}
    with driver.session() as session:
        records = session.run(
            "MATCH (n) WHERE n.uid IN $uids "
            "RETURN n.uid AS uid, n.title AS title, n.content AS content, "
            "labels(n)[0] AS label, n.doc_identity AS doc_identity",
            uids=uids,
        )
        return {row["uid"]: dict(row) for row in records}


def _fetch_documents_sync(driver, identities: list[str]) -> dict[str, dict]:
    """Fetch doc_name and effect_date by doc_identity."""
    if not identities:
        return {}
    with driver.session() as session:
        records = session.run(
            "MATCH (d:Document) WHERE d.doc_identity IN $identities "
            "RETURN d.doc_identity AS doc_identity, d.doc_name AS doc_name, d.effect_date AS effect_date",
            identities=identities,
        )
        return {
            row["doc_identity"]: {
                "name": row["doc_name"], 
                "effect_date": row["effect_date"]
            } 
            for row in records if row["doc_name"]
        }


def _fetch_descendants_sync(driver, article_uids: list[str]) -> dict[str, dict]:
    """Fetch all descendants (Clauses, Points) for the given Article UIDs."""
    if not article_uids:
        return {}
    with driver.session() as session:
        records = session.run(
            "UNWIND $article_uids AS a_uid "
            "MATCH (n) WHERE n.uid STARTS WITH a_uid + '::' "
            "RETURN n.uid AS uid, n.title AS title, n.content AS content, "
            "labels(n)[0] AS label, n.doc_identity AS doc_identity",
            article_uids=article_uids,
        )
        return {row["uid"]: dict(row) for row in records}


def _prefix_article_title(title: str, uid: str) -> str:
    """Prepend 'Điều N.' to article title using the article number from the UID."""
    parts = uid.split("::")
    if len(parts) >= 3 and parts[1] == "article":
        return f"Điều {parts[2]}. {title}"
    return title


def _prefix_clause_content(content: str, uid: str) -> str:
    """Prepend clause number to content using the clause number from the UID."""
    parts = uid.split("::")
    if len(parts) >= 5 and parts[1] == "article" and parts[3] == "clause":
        return f"{parts[4]}. {content}"
    return content


def _prefix_point_content(content: str, uid: str) -> str:
    """Prepend point letter to content using the letter from the UID."""
    parts = uid.split("::")
    if len(parts) >= 7 and parts[1] == "article" and parts[3] == "clause" and parts[5] == "point":
        return f"{parts[6]}) {content}"
    return content


def _build_law_text(
    node: dict,
    parent_article: dict | None,
    parent_clause: dict | None,
    doc_name: str | None,
    all_nodes: dict[str, dict] | None = None,
) -> str:
    """Assemble law text in standard Vietnamese legal document format.

    - Article: Điều N. Title + content
    - Clause:  Điều N. Title + N. clause content
    - Point:   Điều N. Title + N. clause content + đ) point content

    Content that is duplicated at parent level is skipped to avoid repetition.
    doc_name is prepended as the document title (e.g. "Nghị định 168 Về trật tự...").
    """
    uid = node.get("uid", "")
    label = node.get("label", "")
    content = node.get("content", "")

    if label == "Article" and uid.split("::")[-2] == "article":
        title = _prefix_article_title((node.get("title") or "").strip(), uid)
        content = (content or "").strip()
        doc_name = (doc_name or "").strip()
        parts = []
        if doc_name:
            parts.append(doc_name)
        if title:
            parts.append(title)

        # Basic dedup: if content is exactly one of the clauses (often true if only 1 clause exists),
        # we'll still append it if it's there, but usually intro text goes here.
        if content:
            parts.append(content)

        # Include all clauses and points content
        if all_nodes:
            # Clauses have 5 parts in UID: {doc}::article::{N}::clause::{M}
            clauses = [
                n for u, n in all_nodes.items()
                if u.startswith(uid + "::clause::") and len(u.split("::")) == 5
            ]

            def get_clause_num(n):
                p = n.get("uid", "").split("::")
                if len(p) >= 5:
                    try: return int(p[4])
                    except: return p[4]
                return 0
            clauses.sort(key=get_clause_num)

            for clause in clauses:
                c_uid = clause.get("uid", "")
                c_content = (clause.get("content") or "").strip()
                if c_content:
                    parts.append(_prefix_clause_content(c_content, c_uid))

                # Points have 7 parts: {doc}::article::{N}::clause::{M}::point::{L}
                points = [
                    n for u, n in all_nodes.items()
                    if u.startswith(c_uid + "::point::") and len(u.split("::")) == 7
                ]
                # Sort points alphabetically by the last part (letter)
                points.sort(key=lambda n: n.get("uid", "").split("::")[-1])
                for pt in points:
                    p_uid = pt.get("uid", "")
                    p_content = (pt.get("content") or "").strip()
                    if p_content:
                        parts.append(_prefix_point_content(p_content, p_uid))

        return "\n".join(parts)

    # Pre-numbered title for clauses/points
    article_title_raw = (parent_article.get("title") or "") if parent_article else ""
    article_title = _prefix_article_title(article_title_raw, parent_article.get("uid", "")) if parent_article else ""
    article_content = (parent_article.get("content") or "") if parent_article else ""

    if label == "Clause":
        clause_text = _prefix_clause_content(content, uid) if content else ""
        if clause_text:
            parts = [doc_name] if doc_name else []
            if article_title:
                parts.append(article_title)
            parts.append(clause_text)
            return "\n".join(parts)
        if article_title:
            parts = [doc_name] if doc_name else []
            parts.append(article_title)
            if article_content:
                parts.append(article_content)
            return "\n".join(parts)
        return content

    # Point
    point_text = _prefix_point_content(content, uid) if content else ""
    clause_content = (parent_clause.get("content") or "") if parent_clause else ""
    clause_prefixed = _prefix_clause_content(clause_content, parent_clause.get("uid", "")) if clause_content else ""

    # Skip clause content if it's identical to article content (dedup)
    if clause_content == article_content:
        clause_prefixed = ""

    parts = [doc_name] if doc_name else []
    if article_title:
        parts.append(article_title)
    if clause_prefixed:
        parts.append(clause_prefixed)
    if point_text:
        parts.append(point_text)
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict]) -> dict:
    """Compute accuracy, precision, recall, avg_latency_ms across all rows."""
    total = len(rows)
    if total == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0,
                "avg_latency_ms": 0.0, "total_questions": 0, "total_uids": 0,
                "tp": 0, "fp": 0, "tn": 0, "fn": 0}

    # Coerce to bool in case CSV loaded them as strings
    def to_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return bool(v)

    accuracy = sum(1 for r in rows if to_bool(r["verdict"]) == to_bool(r["is_correct"])) / total

    tp = sum(1 for r in rows if to_bool(r["verdict"]) and to_bool(r["is_correct"]))
    fp = sum(1 for r in rows if to_bool(r["verdict"]) and not to_bool(r["is_correct"]))
    tn = sum(1 for r in rows if not to_bool(r["verdict"]) and not to_bool(r["is_correct"]))
    fn = sum(1 for r in rows if not to_bool(r["verdict"]) and to_bool(r["is_correct"]))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    n_questions = len({r["id"] for r in rows})
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "avg_latency_ms": avg_latency,
        "total_questions": n_questions,
        "total_uids": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def compute_model_metrics(rows: list[dict], model: str) -> dict:
    """Per-model metrics: accuracy, precision, recall, avg_latency_ms."""
    model_rows = [r for r in rows if r.get("model") == model]
    total = len(model_rows)
    if total == 0:
        available = sorted({r.get("model") for r in rows})
        sample = rows[0] if rows else {}
        log.debug(
            "compute_model_metrics(%r): no rows. available models: %s  "
            "sample verdict=%r(%s) is_correct=%r(%s)",
            model, available,
            sample.get("verdict"), type(sample.get("verdict")).__name__,
            sample.get("is_correct"), type(sample.get("is_correct")).__name__,
        )
        return {"model": model, "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
                "avg_latency_ms": 0.0, "total_uids": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}

    # Coerce to bool in case CSV loaded them as strings
    def to_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return bool(v)

    # Debug: show first-row types once
    sample = model_rows[0]
    log.debug(
        "compute_model_metrics(%r) sample verdict=%r(%s) is_correct=%r(%s) total=%d",
        model,
        sample.get("verdict"), type(sample.get("verdict")).__name__,
        sample.get("is_correct"), type(sample.get("is_correct")).__name__,
        total,
    )

    accuracy = sum(1 for r in model_rows if to_bool(r["verdict"]) == to_bool(r["is_correct"])) / total

    tp = sum(1 for r in model_rows if to_bool(r["verdict"]) and to_bool(r["is_correct"]))
    fp = sum(1 for r in model_rows if to_bool(r["verdict"]) and not to_bool(r["is_correct"]))
    tn = sum(1 for r in model_rows if not to_bool(r["verdict"]) and not to_bool(r["is_correct"]))
    fn = sum(1 for r in model_rows if not to_bool(r["verdict"]) and to_bool(r["is_correct"]))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    latencies = [r["latency_ms"] for r in model_rows if r.get("latency_ms") is not None]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0

    return {
        "model": model,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "avg_latency_ms": avg_latency,
        "total_uids": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def assemble_merged_law_text(uids: list[str], nodes: dict[str, dict], doc_names: dict[str, str]) -> str:
    """Assembles a single clean text from multiple UIDs, avoiding duplicate headers.

    Groups by Law -> Article -> Clause -> Point.
    Matches hierarchy in _build_law_text for Articles (includes all descendants).
    """
    if not uids:
        return ""

    # Group by Law -> Article -> Clause -> Point
    hierarchy: dict[str, dict] = {}
    for uid in uids:
        parts = uid.split("::")
        doc_id = parts[0]
        if doc_id not in hierarchy:
            hierarchy[doc_id] = {}

        if len(parts) >= 3 and parts[1] == "article":
            art_id = parts[2]
            if art_id not in hierarchy[doc_id]:
                hierarchy[doc_id][art_id] = {}

            if len(parts) >= 5 and parts[3] == "clause":
                cl_id = parts[4]
                if cl_id not in hierarchy[doc_id][art_id]:
                    hierarchy[doc_id][art_id][cl_id] = []

                if len(parts) >= 7 and parts[5] == "point":
                    pt_letter = parts[6]
                    if pt_letter not in hierarchy[doc_id][art_id][cl_id]:
                        hierarchy[doc_id][art_id][cl_id].append(pt_letter)
                else:
                    if None not in hierarchy[doc_id][art_id][cl_id]:
                        hierarchy[doc_id][art_id][cl_id].append(None)
            else:
                if None not in hierarchy[doc_id][art_id]:
                    hierarchy[doc_id][art_id][None] = []

    def _sort_key(x):
        try:
            return int(x)
        except (ValueError, TypeError):
            return x or ""

    final_output = []
    for doc_id in sorted(hierarchy.keys()):
        doc_header = (doc_names.get(doc_id) or "").strip()
        doc_articles = []

        articles = hierarchy[doc_id]
        for art_id in sorted(articles.keys(), key=_sort_key):
            art_lines = []
            art_uid = f"{doc_id}::article::{art_id}"
            art_node = nodes.get(art_uid)
            if not art_node:
                continue

            art_title = _prefix_article_title((art_node.get("title") or "").strip(), art_uid)
            art_lines.append(art_title)

            if None in articles[art_id]:
                # Article itself requested -> show content + ALL descendants
                art_c = (art_node.get("content") or "").strip()
                if art_c:
                    art_lines.append(art_c)

                # All clauses from nodes
                art_prefix = art_uid + "::clause::"
                sub_cl_uids = sorted(
                    [u for u in nodes if u.startswith(art_prefix) and len(u.split("::")) == 5],
                    key=lambda u: _sort_key(u.split("::")[4])
                )
                for cuid in sub_cl_uids:
                    cn = nodes[cuid]
                    cc = (cn.get("content") or "").strip()
                    if cc:
                        art_lines.append(_prefix_clause_content(cc, cuid))

                    pt_prefix = cuid + "::point::"
                    sub_pt_uids = sorted(
                        [u for u in nodes if u.startswith(pt_prefix) and len(u.split("::")) == 7],
                        key=lambda u: u.split("::")[-1]
                    )
                    for puid in sub_pt_uids:
                        pn = nodes[puid]
                        pc = (pn.get("content") or "").strip()
                        if pc:
                            art_lines.append(_prefix_point_content(pc, puid))
            else:
                # Selective clauses/points
                clauses = articles[art_id]
                for cl_id in sorted(clauses.keys(), key=_sort_key):
                    cl_uid = f"{art_uid}::clause::{cl_id}"
                    cl_node = nodes.get(cl_uid)
                    if not cl_node:
                        continue

                    cc = (cl_node.get("content") or "").strip()
                    if cc:
                        art_lines.append(_prefix_clause_content(cc, cl_uid))

                    if None in clauses[cl_id]:
                        # Clause itself requested -> include ALL points under it
                        pt_prefix = cl_uid + "::point::"
                        sub_pt_uids = sorted(
                            [u for u in nodes if u.startswith(pt_prefix) and len(u.split("::")) == 7],
                            key=lambda u: u.split("::")[-1]
                        )
                        for puid in sub_pt_uids:
                            pn = nodes[puid]
                            pc = (pn.get("content") or "").strip()
                            if pc:
                                art_lines.append(_prefix_point_content(pc, puid))
                    else:
                        # Only explicitly requested points
                        pt_letters = sorted([p for p in clauses[cl_id] if p is not None])
                        for pt_letter in pt_letters:
                            puid = f"{cl_uid}::point::{pt_letter}"
                            pn = nodes.get(puid)
                            if pn:
                                pc = (pn.get("content") or "").strip()
                                if pc:
                                    art_lines.append(_prefix_point_content(pc, puid))
            
            doc_articles.append("\n".join(art_lines))

        doc_body = "\n\n".join(doc_articles)
        if doc_header:
            final_output.append(f"{doc_header}\n{doc_body}")
        else:
            final_output.append(doc_body)

    return "\n\n".join(final_output)


# ─────────────────────────────────────────────────────────────────────────────
# Incremental CSV writer (append-safe, no duplicates on re-run)
# ─────────────────────────────────────────────────────────────────────────────

def _write_rows_incremental(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Append rows to a CSV (call site guarantees no duplicates)."""
    if not rows:
        return
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Reusable Law Text Fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_law_texts(embedder, uids: list[str], batch_size: int = 10) -> tuple[dict[str, str], dict[str, dict], str, str]:
    """Fetches law nodes from Neo4j and assembles context-aware law text strings.

    Returns:
        (uid_to_text, all_nodes_map, merged_text, extra_info)
    """
    if not uids:
        return {}, {}, "", ""
    
    from legal_scraper.embedder import Neo4jEmbedder
    driver = embedder._get_driver()

    # 1. Collect all parent UIDs needed for Article/Clause context
    all_needed: set[str] = set()
    for uid in uids:
        all_needed.add(uid)
        pu = _parent_article_uid(uid)
        if pu and pu != uid:
            all_needed.add(pu)
        cu = _parent_clause_uid(uid)
        if cu and cu != uid:
            all_needed.add(cu)

    # 2. Batch-fetch all nodes in one go
    nodes: dict[str, dict] = {}
    for batch_start in range(0, len(all_needed), batch_size):
        batch = list(all_needed)[batch_start:batch_start + batch_size]
        nodes.update(_fetch_nodes_sync(driver, batch))

    # 2.5 Fetch all descendants for Article/Clause nodes among requested UIDs
    parent_uids = [
        uid for uid in uids
        if nodes.get(uid, {}).get("label") in ["Article", "Clause"]
    ]
    if parent_uids:
        nodes.update(_fetch_descendants_sync(driver, parent_uids))

    # 3. Resolve doc_name per unique doc_identity
    identities = list({
        nodes[u].get("doc_identity")
        for u in all_needed
        if nodes.get(u, {}).get("doc_identity")
    })
    doc_meta: dict[str, dict] = {}
    for batch_start in range(0, len(identities), batch_size):
        batch = identities[batch_start:batch_start + batch_size]
        doc_meta.update(_fetch_documents_sync(driver, batch))

    # 4. Build law texts
    import re
    def get_short_name(full_name, identity):
        type_match = re.match(r"^(Luật|Nghị định|Thông tư|Nghị quyết|Quyết định|Sắc lệnh|Sắc luật|Hiến pháp)", full_name, re.IGNORECASE)
        return f"{type_match.group(1)} {identity}" if type_match else identity

    uid_to_text: dict[str, str] = {}
    doc_names = {i: get_short_name(d["name"], i) for i, d in doc_meta.items()}
    for uid in uids:
        node = nodes.get(uid, {})
        article_uid = _parent_article_uid(uid)
        clause_uid = _parent_clause_uid(uid)
        parent_article = nodes.get(article_uid) if article_uid and article_uid in nodes else None
        parent_clause = nodes.get(clause_uid) if clause_uid and clause_uid in nodes else None
        doc_identity = node.get("doc_identity", "")
        doc_name = doc_names.get(doc_identity) if doc_identity else None
        uid_to_text[uid] = _build_law_text(node, parent_article, parent_clause, doc_name, nodes)

    # 5. Build single merged text (deduplicated headers)
    merged_text = assemble_merged_law_text(uids, nodes, doc_names)

    # 6. Extra metadata (amends and effect dates)
    amends_map = embedder.fetch_amends(uids)
    amends_info = embedder.format_amends(amends_map, uids)
    
    date_lines = []
    for d_id in sorted(doc_meta.keys()):
        m = doc_meta[d_id]
        eff = m["effect_date"]
        eff_str = f" (Ngày hiệu lực: {eff[:10]})" if eff else ""
        short_name = doc_names.get(d_id, d_id)
        date_lines.append(f"- {short_name}{eff_str}")
    
    extra_info = ""
    if date_lines:
        extra_info += "Danh sách văn bản hiệu lực:\n" + "\n".join(date_lines) + "\n"
    if amends_info:
        extra_info += "\nThông tin sửa đổi:\n" + amends_info + "\n"

    return uid_to_text, nodes, merged_text, extra_info


def parse_eval_score(raw: str) -> int | None:
    """Extracts 'Score: [N]' or 'Score: N' from LLM response, handling bolding markers."""
    # Look for Score: [0-5], **Score:** 5, etc.
    # Pattern: 'Score' -> optional colon/stars -> optional brackets -> digit
    match = re.search(r"Score[:\s*]*\**\s*\[?(\d)\]?", raw, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


async def evaluate_qa(
    input_path: str,
    output_dir: str,
    prompt_template_path: str,
    backends: list,
    model_names: list[str],
    gt_dataset_path: str | None = None,
    payload_dir: str | None = None,
    dataset_path: str | None = None,
    print_every: int = 5,
    start_index: int = 0,
) -> None:
    """Evaluates QA answers using a single LLM as a judge."""
    df = pd.read_csv(input_path)
    
    # 1. Merge with Offline Payload if provided (for full context)
    def clean_id(s):
        s = str(s).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return s

    if payload_dir and dataset_path:
        p_path = get_payload_path(dataset_path, payload_dir)
        if p_path.exists():
            log.info("Loading full context from payload: %s", p_path)
            payload_data = {}
            with open(p_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    pid = clean_id(item["id"])
                    payload_data[pid] = {
                        "law_text": item.get("law_text", ""),
                        "extra_info": item.get("extra_info", ""),
                    }
            
            # Map payload data back to df
            df["id_str"] = df["id"].apply(clean_id)
            df["law_text_full"] = df["id_str"].map(lambda x: payload_data.get(x, {}).get("law_text", ""))
            df["extra_info_full"] = df["id_str"].map(lambda x: payload_data.get(x, {}).get("extra_info", ""))
            
            # Use full text from payload, fallback to existing columns ONLY if payload is empty
            df["law_text_preview"] = df["law_text_full"].where(df["law_text_full"] != "", df["law_text_preview"])
            df["extra_info"] = df["extra_info_full"].where(df["extra_info_full"] != "", df.get("extra_info", ""))
            
            df.drop(columns=["id_str", "law_text_full", "extra_info_full"], inplace=True)
        else:
            log.warning("Payload file not found at %s", p_path)

    # 2. Merge with Ground Truth dataset if provided
    if gt_dataset_path:
        log.info("Merging with Ground Truth dataset: %s", gt_dataset_path)
        gt_df = pd.read_csv(gt_dataset_path)
        
        # Robust ID matching: strip and convert to string, remove .0 from numeric IDs
        def clean_id(s):
            s = str(s).strip()
            if s.endswith(".0"):
                s = s[:-2]
            return s

        df["id"] = df["id"].apply(clean_id)
        gt_df["id"] = gt_df["id"].apply(clean_id)
        
        # Rename 'answer' to 'expert_answer' in GT df if it exists
        if "answer" in gt_df.columns and "expert_answer" not in gt_df.columns:
            gt_df = gt_df.rename(columns={"answer": "expert_answer"})
            
        # We want GT to be the STRICT source for expert_answer
        if "expert_answer" in df.columns and "expert_answer" in gt_df.columns:
            df.drop(columns=["expert_answer"], inplace=True)

        # Merge on 'id', keeping only necessary GT columns (ignore extra_info from GT as we want it from payload)
        gt_cols = ["id", "expert_answer"]
        if "question" in gt_df.columns and "question" not in df.columns:
            gt_cols.append("question")
            
        df = df.merge(gt_df[gt_cols], on="id", how="left")
        
        # Verify if ground truth was successfully merged
        missing_gt = df["expert_answer"].isna().sum()
        if missing_gt > 0:
            log.warning("Ground truth missing for %d rows after merge", missing_gt)

    if start_index > 0:
        df = df.iloc[start_index:]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    template = Path(prompt_template_path).read_text(encoding="utf-8")
    backend = backends[0]
    eval_model_name = model_names[0]
    
    results_path = output_dir / "row_qa_eval_scores.csv"
    fieldnames = ["id", "question", "score", "reasoning", "comparison", "raw_eval", "latency_ms"]
    
    if not results_path.exists() or start_index == 0:
        with open(results_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    all_scores = []
    log.info("Starting QA Evaluation using judge model: %s", eval_model_name)

    from tqdm.auto import tqdm
    for idx, row in enumerate(tqdm(df.itertuples(), total=len(df), desc="QA Eval"), 1):
        qid = str(getattr(row, "id", idx + start_index))
        question = str(getattr(row, "question", ""))
        
        # Helper to safely get values and avoid 'nan' strings
        def get_val(attr, default=""):
            v = getattr(row, attr, default)
            return str(v) if pd.notna(v) and v is not None else default

        expert_answer = get_val("expert_answer", "")
        llm_answer = get_val("generated_answer", get_val("raw_response", ""))
        law_text = get_val("law_text_preview", "")
        extra_info = get_val("extra_info", "")

        # Build prompt
        # The template has: {law_text}, {extra_info}, {question}, {ground_truth}, {llm_answer}
        prompt = template.format(
            question=question,
            law_text=law_text,
            extra_info=f"Thông tin bổ sung: \n{extra_info}" if extra_info else "",
            ground_truth=expert_answer,
            llm_answer=llm_answer
        )

        # Apply Gemma token fixes if needed
        if "gemma" in eval_model_name.lower():
            # Specific mappings for Gemma 3 / Fine-tuned templates
            prompt = prompt.replace("<|im_start|>user", "<|turn>user")
            prompt = prompt.replace("<|im_start|>assistant", "<|turn>model")
            # Fallback / General tokens
            prompt = prompt.replace("<|im_start|>", "<|turn>")
            prompt = prompt.replace("<|im_end|>", "<turn|>")
            # Thinking / Reasoning tokens
            prompt = prompt.replace("<think>", "<|channel>thought")
            prompt = prompt.replace("</think>", "<channel|>")
        

        # Retry logic for empty or failed responses
        max_retries = 3
        raw_eval = ""
        duration_ms = 0
        score = None
        
        for attempt in range(max_retries):
            start_t = time.perf_counter()
            try:
                raw_eval = await backend.ask(prompt)
                duration_ms = (time.perf_counter() - start_t) * 1000
                # print('prompt', prompt)
                # print('answer', raw_eval)
                if raw_eval and raw_eval.strip():
                    score = parse_eval_score(raw_eval)
                    break
                else:
                    log.warning("Empty evaluation response from judge (attempt %d/%d)", attempt+1, max_retries)
            except Exception as e:
                log.error("Judge failed for qid=%s (attempt %d/%d): %s", qid, attempt+1, max_retries, e)
                raw_eval = f"ERROR: {e}"
                duration_ms = (time.perf_counter() - start_t) * 1000
            
            if attempt < max_retries - 1:
                await asyncio.sleep(1)

        is_pass = (score >= 3) if score is not None else False
        if score is not None:
            all_scores.append(score)

        # Basic parsing for reasoning/comparison/score using robust regex
        reasoning = ""
        comparison = ""
        
        # Regex to find sections: - **Field:** Content
        r_match = re.search(r"Reasoning[:\s*]*\**\s*(.*?)(?=- \*\*|- Score:|$)", raw_eval, re.DOTALL | re.IGNORECASE)
        if r_match:
            reasoning = r_match.group(1).strip().strip(":")
            
        c_match = re.search(r"Comparison[:\s*]*\**\s*(.*?)(?=- \*\*|- Score:|$)", raw_eval, re.DOTALL | re.IGNORECASE)
        if c_match:
            comparison = c_match.group(1).strip().strip(":")

        result_row = {
            "id": qid,
            "question": question,
            "score": score,
            "reasoning": reasoning[:500],
            "comparison": comparison[:500],
            "raw_eval": raw_eval[:1000],
            "latency_ms": round(duration_ms, 1)
        }
        
        _write_rows_incremental(results_path, [result_row], fieldnames)

    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        pass_rate = sum(1 for s in all_scores if s >= 3) / len(all_scores)
        log.info("QA Eval Summary: Avg Score = %.2f, Pass Rate (>=3) = %.2f%%", avg_score, pass_rate * 100)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

async def evaluate(
    input_path: str,
    output_dir: str,
    uri: str,
    user: str,
    password: str,
    database: str,
    top_k: int = 5,
    prompt_template: str | None = None,
    models: list[str] | None = None,
    batch_size: int = 10,
    backend: str = "llama_cpp",
    n_voters: int = 3,
    model: str | None = None,
    base_url: str | None = None,
    base_port: int = 8000,
    api_key: str = "vllm-secret-key",
    n_gpu_layers: int = 0,
    n_ctx: int = 4096,
    print_every: int = 5,
    start_index: int = 0,
    mode: str = "classify",
    gt_dataset: str | None = None,
    payload_dir: str | None = None,
    dataset_path: str | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Backends
    voter_models = []
    if models is not None:
        if len(models) != n_voters:
            raise ValueError(f"--models ({len(models)}) must match --n-voters ({n_voters})")
        voter_models = models
    elif model is not None:
        voter_models = [model] * n_voters
    elif backend == "llama_cpp":
        voter_models = [
            "/content/drive/MyDrive/HCMUS/NLP-LegalQA/models/Qwen3-4B-Q4_K_M.gguf",
            "/content/drive/MyDrive/HCMUS/NLP-LegalQA/models/nanbeige4.1-Q4_K_M.gguf",
        ] * (n_voters // 2 + 1)
        voter_models = voter_models[:n_voters]
    else:
        raise ValueError("Must provide either --model or --models for non-llama_cpp backends")

    _backends = []
    if backend == "llama_cpp":
        from voter import LlamaCppBackend
        for m in voter_models:
            _backends.append(LlamaCppBackend(model_path=m, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx))
        model_names = [Path(m).stem for m in voter_models]
    elif backend == "vllm":
        from voter import VLLMBackend
        for i, m in enumerate(voter_models):
            port = base_port + i
            _backends.append(VLLMBackend(model=m, base_url=f"http://localhost:{port}/v1", api_key=api_key))
        model_names = voter_models
    else:  # ollama
        from voter import OllamaBackend
        _url = base_url or "http://localhost:11434"
        for m in voter_models:
            _backends.append(OllamaBackend(model=m, base_url=_url))
        model_names = voter_models

    # 2. Dispatch to Mode
    if mode == "eval_qa":
        await evaluate_qa(
            input_path=input_path,
            output_dir=output_dir,
            prompt_template_path=prompt_template or "src/llm/prompt_eval_qa.md",
            backends=_backends,
            model_names=model_names,
            gt_dataset_path=gt_dataset,
            payload_dir=payload_dir,
            dataset_path=dataset_path,
            print_every=print_every,
            start_index=start_index
        )
        return

    # Default: classify mode (Voter Relevance)
    from voter import LegalVoter
    from legal_scraper.embedder import Neo4jEmbedder
    voter = LegalVoter(backends=_backends, model_names=model_names)
    
    df = pd.read_csv(input_path)
    if start_index > 0:
        df = df.iloc[start_index:]
    log.info("Loaded dataset. Mode: classify. Processing from index %d (%d rows)", start_index, len(df))

    embedder = Neo4jEmbedder(uri=uri, user=user, password=password, database=database)

    # Collect distinct model names now (needed inside the loop for incremental print)
    all_model_names: list[str] = list(dict.fromkeys(model_names))

    all_rows: list[dict] = []
    uid_fieldnames = [
        "id", "question", "uid", "label", "is_correct", "verdict",
        "model", "latency_ms", "raw_response", "law_text_preview",
    ]
    row_results_path = output_dir / "row_voter_results.csv"
    # Pre-create / wipe to write fresh headers on re-run
    if not row_results_path.exists() or start_index == 0:
        with open(row_results_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=uid_fieldnames).writeheader()
    
    last_save_len = 0  # track how many rows were saved; only append new ones
    driver = embedder._get_driver()

    with _swap_prompt_template(prompt_template):
        n_questions = len(df)
        for q_idx, row in enumerate(df.itertuples(), 1):
            qid = str(row.id if hasattr(row, "id") else q_idx)
            question = str(row.question)

            # Parse references
            raw_refs = str(getattr(row, "references", "")).strip()
            references = [r.strip() for r in raw_refs.split(";") if r.strip()]

            # Parse retrieved UIDs (top-k)
            raw_uids = str(getattr(row, "retrieved_uids", "")).strip()
            retrieved_uids = [u.strip() for u in raw_uids.split(";") if u.strip()][:top_k]

            if not retrieved_uids:
                log.warning("No retrieved UIDs for question %s — skipping", qid)
                continue

            # Fetch law texts and metadata for all retrieved UIDs
            uid_to_law_text, nodes, _, _ = fetch_law_texts(embedder, retrieved_uids, batch_size=batch_size)

            for uid in retrieved_uids:
                law_text = uid_to_law_text.get(uid, "")
                node = nodes.get(uid, {})

                # Vote
                try:
                    result: VoteResult = await voter.vote(question, law_text)
                except Exception as e:
                    log.error("Voter failed for uid=%s: %s", uid, e)
                    result = VoteResult(verdict=False)

                # Ground-truth relevance
                correct = any(is_relevant(uid, ref) for ref in references)
                total_duration_ms = result.total_duration_ms

                # Per-model rows (one per voter)
                for vote, model_name, raw, elapsed_ms in zip(
                    result.votes, result.models, result.raw_responses, result.vote_durations_ms
                ):
                    all_rows.append({
                        "id": qid,
                        "question": question,
                        "uid": uid,
                        "label": node.get("label", ""),
                        "is_correct": correct,
                        "verdict": vote,
                        "model": model_name,
                        "latency_ms": round(elapsed_ms, 1),
                        "raw_response": raw[:300] if raw else "",
                        "law_text_preview": law_text[:200] if law_text else "",
                    })

                # Majority-verdict row (aggregate)
                all_rows.append({
                    "id": qid,
                    "question": question,
                    "uid": uid,
                    "label": node.get("label", ""),
                    "is_correct": correct,
                    "verdict": result.verdict,
                    "model": "MAJORITY",
                    "latency_ms": round(total_duration_ms, 1) if total_duration_ms else None,
                    "raw_response": "",
                    "law_text_preview": law_text[:200] if law_text else "",
                })

                # Per-uid log
                tag = "TP" if (result.verdict and correct) else \
                      "FP" if (result.verdict and not correct) else \
                      "FN" if (not result.verdict and correct) else "TN"
                log.debug("[%s] uid=%s majority=%s votes=%s", tag, uid, result.verdict, result.votes)

            # Incremental print every N UIDs
            uid_count = len(all_rows)
            if print_every > 0 and uid_count % print_every == 0:
                # Derive model names from what's actually in all_rows (avoids mismatch with backend definitions)
                all_model_names_snapshot = list(dict.fromkeys(
                    r["model"] for r in all_rows
                    if r["model"] not in ("MAJORITY", "")
                ))
                snapshot = [r for r in all_rows if r["model"] == "MAJORITY"]
                snap_metrics = compute_metrics(snapshot)
                snap_model_rows = [
                    compute_model_metrics(all_rows, m)
                    for m in all_model_names_snapshot
                ]
                print(f"\n=== Incremental @ Q{q_idx}/{n_questions} | {uid_count} UIDs ===")
                print(f"  Majority  acc={snap_metrics['accuracy']:.4f}  "
                      f"prec={snap_metrics['precision']:.4f}(tp={snap_metrics['tp']},fp={snap_metrics['fp']})  "
                      f"rec={snap_metrics['recall']:.4f}(tp={snap_metrics['tp']},fn={snap_metrics['fn']})  "
                      f"lat={snap_metrics['avg_latency_ms']:.1f}ms")
                for m in snap_model_rows:
                    print(f"  {m['model']:<30} acc={m['accuracy']:.4f}  "
                          f"prec={m['precision']:.4f}(tp={m['tp']},fp={m['fp']},tn={m['tn']},fn={m['fn']})  "
                          f"rec={m['recall']:.4f}  "
                          f"lat={m['avg_latency_ms']:.1f}ms  n={m['total_uids']}")
                # Save only the newly accumulated rows
                new_rows = all_rows[last_save_len:]
                _write_rows_incremental(row_results_path, new_rows, uid_fieldnames)
                last_save_len = len(all_rows)
                print(f"  [saved {uid_count} UIDs to {row_results_path.name}]")
                print()

    driver.close()

    # ── Final CSV write — append any rows not yet checkpointed ─────────────────
    _write_rows_incremental(row_results_path, all_rows, uid_fieldnames)
    log.info("Final write: total rows in CSV = %s", sum(1 for _ in open(row_results_path)) - 1)

    # Derive model names from all_rows (in case backends returned different names)
    final_model_names = list(dict.fromkeys(
        r["model"] for r in all_rows
        if r["model"] not in ("MAJORITY", "")
    ))

    # ── Per-model CSV ──────────────────────────────────────────────────────────
    model_metrics_path = output_dir / "voter_model_metrics.csv"
    model_rows_out = [compute_model_metrics(all_rows, m) for m in final_model_names]
    with open(model_metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "precision", "recall",
                                                  "avg_latency_ms", "total_uids",
                                                  "tp", "fp", "tn", "fn"])
        writer.writeheader()
        writer.writerows(model_rows_out)
    log.info("Wrote per-model metrics → %s", model_metrics_path)

    # ── Overall summary CSV ────────────────────────────────────────────────────
    metrics = compute_metrics(all_rows)
    summary_path = output_dir / "voter_metrics_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    log.info("Wrote summary → %s", summary_path)

    log.info("=== Voter Evaluation Results ===")
    log.info("Accuracy:     %.4f", metrics["accuracy"])
    log.info("Precision:   %.4f  tp=%d fp=%d tn=%d fn=%d",
             metrics["precision"], metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"])
    log.info("Recall:      %.4f  tp=%d fp=%d tn=%d fn=%d",
             metrics["recall"], metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"])
    log.info("Avg latency: %.1f ms", metrics["avg_latency_ms"])
    log.info("Questions:   %d", metrics["total_questions"])
    log.info("Total UIDs:  %d", metrics["total_uids"])
    log.info("--- Per-model ---")
    for m in model_rows_out:
        log.info("  %-30s acc=%.4f  prec=%.4f(tp=%d,fp=%d,tn=%d,fn=%d)  rec=%.4f  lat=%.1fms  n=%d",
                 m["model"], m["accuracy"],
                 m["precision"], m["tp"], m["fp"], m["tn"], m["fn"],
                 m["recall"],
                 m["avg_latency_ms"], m["total_uids"])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate voter on RAG results")
    parser.add_argument("--mode", choices=["classify", "eval_qa"], default="classify", help="Evaluation mode")
    parser.add_argument("--input", required=True, help="Path to input CSV (row_results.csv or row_qa_results_offline.csv)")
    parser.add_argument("--gt-dataset", help="Optional path to ground truth dataset (e.g. QA_NLP.csv) to merge by 'id'")
    parser.add_argument("--payload-dir", help="Directory where offline payloads (.jsonl) are stored")
    parser.add_argument("--dataset", help="Original dataset path (used to find payload filename)")
    parser.add_argument("--output", required=True, help="Output directory")
    # Neo4j
    parser.add_argument("--uri", default="neo4j+ssc://nguyenhoangquan.com:7687", help="Neo4j URI (e.g. neo4j+ssc://host:7687)")
    parser.add_argument("--user", default="neo4j", help="Neo4j user")
    parser.add_argument("--password", default="Neoneo4j", help="Neo4j password")
    parser.add_argument("--database", default="neo4j", help="Neo4j database")
    # Voter options
    parser.add_argument("--top-k", type=int, default=5, help="Top-K retrieved UIDs to evaluate")
    parser.add_argument(
        "--prompt-template",
        default="src/llm/prompt_classify.md",
        help="Path to custom prompt .md template (default: src/llm/prompt_classify.md)",
    )
    parser.add_argument(
        "--backend",
        choices=["llama_cpp", "ollama", "vllm"],
        default="llama_cpp",
        help="LLM backend: 'llama_cpp' (loads GGUF directly, default), 'ollama', or 'vllm'",
    )
    parser.add_argument(
        "--n-voters",
        type=int,
        default=3,
        help="Number of voter LLMs (vllm: voter i at localhost:{base-port+i}, default 3)",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=5,
        help="Print incremental per-model + summary metrics every N UIDs (default 5, use 0 to disable)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Single model name (vllm) or GGUF path (llama_cpp/ollama when n_voters=1)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="GGUF paths (llama_cpp) or model names (ollama). "
             "Length must equal n-voters. Overrides --model.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Server base URL for ollama (default localhost:11434)",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=8000,
        help="Starting port for vllm voters (voter i → localhost:{base-port+i}, default 8000)",
    )
    parser.add_argument(
        "--api-key",
        default="vllm-secret-key",
        help="API key for vllm server auth",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="Number of GPU layers for llama_cpp (0 = CPU, increase for GPU offload; T4: try 20-35)",
    )
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context window size for llama_cpp")
    parser.add_argument("--batch-size", type=int, default=10, help="Neo4j fetch batch size")
    parser.add_argument("--start-index", type=int, default=0, help="Starting row index in the dataset (0-based)")
    args = parser.parse_args()

    # Env var fallbacks
    args.uri = os.environ.get("NEO4J_URI", args.uri)
    args.user = os.environ.get("NEO4J_USER", args.user)
    args.password = os.environ.get("NEO4J_PASSWORD", args.password)
    args.database = os.environ.get("NEO4J_DATABASE", args.database)

    asyncio.run(evaluate(
        input_path=args.input,
        output_dir=args.output,
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
        top_k=args.top_k,
        prompt_template=args.prompt_template,
        models=args.models,
        batch_size=args.batch_size,
        backend=args.backend,
        n_voters=args.n_voters,
        model=args.model,
        base_port=args.base_port,
        api_key=args.api_key,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        print_every=args.print_every,
        start_index=args.start_index,
        mode=args.mode,
        gt_dataset=args.gt_dataset,
        payload_dir=args.payload_dir,
        dataset_path=args.dataset,
    ))


if __name__ == "__main__":
    main()
