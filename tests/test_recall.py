"""
Recall@K evaluation for the DECOMPOSITION pipeline.
Loads ground truth from CSV, runs decompose -> multi_search -> rerank,
computes Recall@K for each question.
"""

from dotenv import load_dotenv
load_dotenv()
import os
import sys
import csv
sys.stdout.reconfigure(encoding="utf-8")

from legal_scraper.embedder import Neo4jEmbedder, SearchResult
from legal_scraper.reranker import VietnameseReranker

# Config
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "qa_dataset", "QA_NLP.csv")
K_VALUES = [3, 5, 10]
TOP_K_PER_SQ = 5
RERANK_TOP_K = 10


def load_ground_truth(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question", "").strip()
            ref_str = row.get("reference", "").strip()
            refs = [r.strip() for r in ref_str.split(",") if r.strip()]
            if q:
                rows.append({"question": q, "references": refs})
    return rows


def recall_at_k(retrieved: list[str], ground_truth: list[str], k: int) -> float:
    top_k = set(retrieved[:k])
    gt = set(ground_truth)
    if not gt:
        return 0.0
    return len(top_k & gt) / len(gt)


def main():
    gt_data = load_ground_truth(CSV_PATH)
    if not gt_data:
        print("No ground truth found. Add references to qa_dataset/QA_NLP.csv first.")
        return

    print(f"Loaded {len(gt_data)} questions from CSV\n")

    e = Neo4jEmbedder(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"],
        database=os.environ.get("NEO4J_DATABASE", "neo4j"),
        openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
    )
    reranker = VietnameseReranker(device="cpu")

    recall = {k: [] for k in K_VALUES}

    for i, row in enumerate(gt_data):
        question = row["question"]
        gt_uids = row["references"]

        print(f"[{i+1}/{len(gt_data)}] Q: {question}")
        print(f"  Ground truth: {gt_uids}")

        # Step 1: decompose
        decomp = e.decompose_query_debug(question)
        if not decomp.sub_queries:
            print("  Decompose FAILED — skipping")
            continue

        print(f"  Sub-queries: {[sq['query'] for sq in decomp.sub_queries]}")

        # Step 2: multi_search
        sq_results = e.multi_search(decomp.sub_queries, k=TOP_K_PER_SQ)

        # Collect all UIDs across sub-queries (order-preserving)
        seen, all_uids = set(), []
        for idx, hits in sq_results.items():
            for r in hits:
                if r.uid not in seen:
                    seen.add(r.uid)
                    all_uids.append(r.uid)

        print(f"  Total hits from multi_search: {len(all_uids)}")

        # Step 3: fetch content
        all_labels = list(set(r.label for hits in sq_results.values() for r in hits))
        node_contents = e.fetch_nodes(all_uids, all_labels)

        # Step 4: rerank
        docs, uid_list = [], []
        for uid in all_uids[:RERANK_TOP_K]:
            for label in all_labels:
                key = (uid, label)
                if key in node_contents:
                    content = node_contents[key].get("content", "")
                    title = node_contents[key].get("title") or ""
                    text = (f"{title}\n{content}" if title else content).strip()
                    docs.append(text)
                    uid_list.append(uid)
                    break

        if docs:
            reranked = reranker.rerank(question, docs, top_k=len(docs), batch_size=4)
            final_uids = [uid_list[i] for i, _ in reranked]
        else:
            final_uids = all_uids[:max(K_VALUES)]

        print(f"  Top-5 after rerank: {final_uids[:5]}")

        # Step 5: Recall@K
        for k in K_VALUES:
            r = recall_at_k(final_uids, gt_uids, k)
            recall[k].append(r)
            print(f"  Recall@{k}: {r:.2%}")

        print()

    # Summary
    print("=" * 50)
    print(f"{'K':<6} {'Recall@K':>12} {'Count':>8}")
    print("-" * 50)
    for k in K_VALUES:
        avg = sum(recall[k]) / len(recall[k]) if recall[k] else 0
        print(f"  @{k:<4} {avg:>10.2%}  {len(recall[k]):>6} questions")
    print("=" * 50)

    e.close()


if __name__ == "__main__":
    main()
