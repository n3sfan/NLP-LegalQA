from collections import namedtuple
import os
from dataclasses import dataclass
from typing import List, Optional
import json

from neo4j import GraphDatabase

SearchResult = namedtuple("SearchResult", ["uid", "label", "score"])


class VietnameseEmbeddings:
    """Custom LangChain Embeddings wrapper for Vietnamese word segmentation.

    Wraps :class:`langchain_huggingface.HuggingFaceEmbeddings` and
    automatically applies :func:`pyvi.ViTokenizer.tokenize` (underscore-style)
    before embedding both documents and queries.
    """

    def __init__(self, model_name: str = "bkai-foundation-models/vietnamese-bi-encoder", **model_kwargs):
        from langchain_huggingface import HuggingFaceEmbeddings

        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            **model_kwargs,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Tokenize each document and embed via the underlying model."""
        from pyvi.ViTokenizer import tokenize
        tokenized = [tokenize(text) for text in texts]
        return self._embeddings.embed_documents(tokenized)

    def embed_query(self, text: str) -> List[float]:
        """Tokenize the query and embed via the underlying model."""
        from pyvi.ViTokenizer import tokenize
        return self._embeddings.embed_query(tokenize(text))



class Neo4jEmbedder:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = None
        self._embedding_model = None

    def _get_embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = VietnameseEmbeddings(
                model_name="bkai-foundation-models/vietnamese-bi-encoder",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embedding_model

    def _get_driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver

    def embed_label(self, labels: str | list[str], batch_size: int = 32) -> None:
        """Generate and store embeddings for all nodes of the given label(s).

        Args:
            labels: Single label (e.g. "Article") or list of labels
                    (e.g. ["Article", "Clause", "Point"]).
            batch_size: Batch size hint (passed but not wired through
                        HuggingFaceEmbeddings in this version).
        """
        if isinstance(labels, str):
            labels = [labels]

        from langchain_neo4j import Neo4jVector

        embedding_model = self._get_embedding_model()

        for label in labels:
            Neo4jVector.from_existing_graph(
                embedding=embedding_model,
                url=self.uri,
                username=self.user,
                password=self.password,
                database=self.database,
                index_name=f"{label}_embedding_index",
                node_label=label,
                text_node_properties=["title", "content"] if label == "Article" else ["content"],
                embedding_node_property="embedding",
            )

    def search(
        self,
        labels: str | list[str],
        query: str,
        k: int = 5,
    ) -> list[SearchResult]:
        """Search across all given label(s) and return top-k results sorted by score.

        Args:
            labels: Single label (e.g. "Clause") or list of labels
                    (e.g. ["Article", "Clause", "Point"]). All indexes are
                    queried and results merged.
            query: Raw Vietnamese text query.
            k: Number of results to return per label (default 5).
                Total results may be up to k * len(labels).

        Returns:
            List of SearchResult(uid, label, score), sorted by score descending.
        """
        from langchain_neo4j import Neo4jVector

        if isinstance(labels, str):
            labels = [labels]

        embedding_model = self._get_embedding_model()

        all_results: list[SearchResult] = []

        for label in labels:
            vector = Neo4jVector.from_existing_index(
                embedding=embedding_model,
                url=self.uri,
                username=self.user,
                password=self.password,
                database=self.database,
                index_name=f"{label}_embedding_index",
                text_node_properties=["title", "content"] if label == "Article" else ["content"],
                embedding_node_property="embedding",
            )

            docs = vector.similarity_search_with_score(query, k=k)
            for doc, score in docs:
                uid = doc.metadata.get("uid", "")
                all_results.append(SearchResult(uid=uid, label=label, score=score))

        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results

    def fetch_nodes(self, uids: list[str], labels: list[str]) -> dict[tuple[str, str], dict]:
        """Fetch content and title for nodes by (uid, label).

        Uses a single Cypher query per label.
        Returns a dict keyed by (uid, label) -> {"content": str, "title": str|None}.
        Title is non-null only for Article nodes.
        Missing nodes are silently omitted from the result dict.
        """
        with self._get_driver().session(database=self.database) as session:
            result: dict[tuple[str, str], dict] = {}
            for label in labels:
                records = session.run(
                    f"MATCH (n:{label}) WHERE n.uid IN $uids "
                    "RETURN n.uid AS uid, n.content AS content, n.title AS title",
                    uids=uids,
                )
                for record in records:
                    uid = record["uid"]
                    title = record["title"]
                    result[(uid, label)] = {
                        "content": record["content"] or "",
                        "title": title if title else None,
                    }
            return result

    def fetch_node_hierarchy(self, uids: list[str]) -> dict[str, str]:
        """Fetch nodes and format their parent hierarchy into context text.
        
        Uses a variable-length path query from Document down to the target node
        to reconstruct the full context (e.g., Chapter > Section > Article > Clause).
        The Document node itself is dropped from the returned string, per user request.
        
        Returns a dict mapping uid -> formatted context string.
        """
        query = """
        UNWIND $uids AS target_uid
        MATCH path = (d:Document)-[:HAS_PART|HAS_CHAPTER|HAS_SECTION|HAS_ARTICLE|HAS_CLAUSE|HAS_POINT *]->(target)
        WHERE target.uid = target_uid
        RETURN target.uid AS uid, nodes(path)[1..] AS hierarchy_nodes
        """
        result_map: dict[str, str] = {}
        with self._get_driver().session(database=self.database) as session:
            result = session.run(query, uids=uids)
            for record in result:
                uid = record["uid"]
                hierarchy_nodes = record["hierarchy_nodes"] or []
                
                context_lines = []
                for n in hierarchy_nodes:
                    labels = list(n.labels)
                    label = labels[0] if labels else ""
                    
                    if label == "Part":
                        title = n.get("title")
                        context_lines.append(f"Phần {n.get('number')}: {title}" if title else f"Phần {n.get('number')}")
                    elif label == "Chapter":
                        title = n.get("title")
                        context_lines.append(f"Chương {n.get('number')}: {title}" if title else f"Chương {n.get('number')}")
                    elif label == "Section":
                        title = n.get("title")
                        context_lines.append(f"Mục {n.get('number')}: {title}" if title else f"Mục {n.get('number')}")
                    elif label == "Article":
                        title = n.get("title")
                        context_lines.append(f"Điều {n.get('number')}: {title}" if title else f"Điều {n.get('number')}")
                    elif label == "Clause":
                        if n == hierarchy_nodes[-1]:
                            context_lines.append(f"Khoản {n.get('number')}.")
                        else:
                            clause_content = n.get('content', '').strip()
                            if clause_content:
                                context_lines.append(f"Khoản {n.get('number')}.\n{clause_content}")
                            else:
                                context_lines.append(f"Khoản {n.get('number')}.")
                    elif label == "Point":
                        context_lines.append(f"Điểm {n.get('letter')}.")

                # The last node is the target itself. We can extract its content explicitly 
                # (though it might already have been hit by the loop above, we'll append the detailed content)
                target_node = hierarchy_nodes[-1] if hierarchy_nodes else None
                main_content = target_node.get("content", "") if target_node else ""
                
                # Combine headers and full content
                header_text = "\n".join(context_lines)
                full_text = header_text + "\nNội dung: " + main_content if main_content else header_text
                result_map[uid] = full_text.strip()
                
        return result_map

    @staticmethod
    def format_uid_vn(uid: str) -> str:
        """Format a technical UID into a Vietnamese legal reference.
        
        Example: "168/2024/NĐ-CP::article::52::clause::8::point::d" 
                 -> "Điểm d Khoản 8 Điều 52 168/2024/NĐ-CP"
        """
        if "::" not in uid:
            return uid
            
        parts = uid.split("::")
        doc_id = parts[0]
        segments = parts[1:]
        
        label_map = {
            "article": "Điều",
            "clause": "Khoản",
            "point": "Điểm",
            "section": "Mục",
            "chapter": "Chương",
            "part": "Phần"
        }
        
        formatted_segments = []
        for i in range(0, len(segments), 2):
            if i + 1 >= len(segments):
                break
            label = segments[i].lower()
            val = segments[i+1]
            vn_label = label_map.get(label, label.capitalize())
            formatted_segments.append(f"{vn_label} {val}")
            
        # Reverse to get smaller units first (Point -> Clause -> Article)
        formatted_segments.reverse()
        return " ".join(formatted_segments + [doc_id])

    @staticmethod
    def format_amends(amends_map: dict[str, list[dict]], uids: list[str]) -> str:
        """Format amendment metadata into a readable string (reused from cli.py)."""
        output = []
        for uid in uids:
            amends = amends_map.get(uid, [])
            if amends:
                # User requested to remove the "[!] NOTE:" line from prompt
                for amend in amends:
                    eff_date = amend.get('effect_date')
                    eff_str = f" (Effective: {eff_date[:10]})" if eff_date and len(eff_date) >= 10 else ""
                    
                    amending_vn = Neo4jEmbedder.format_uid_vn(amend['amending_uid'])
                    amended_vn = Neo4jEmbedder.format_uid_vn(amend['amended_uid'])
                    
                    output.append(f"      - Amended by: {amending_vn}{eff_str} (applied to {amended_vn})")
                    output.append(f"        Content: {amend['amending_content']}")
        return "\n".join(output)

    def fetch_amends(self, uids: list[str]) -> dict[str, list[dict]]:
        """Fetch amends for the given node UIDs.
        
        An item is considered amended if it is amended directly or if any of its parent nodes
        (Article, Chapter, etc.) are amended.
        Returns a mapping from target_uid to a list of amendment dictionaries.
        """
        query = """
        UNWIND $uids AS target_uid
        MATCH (target) WHERE target.uid = target_uid
        // Match the target itself, its parents (up to Document), and its descendants
        MATCH (h_node)
        WHERE (h_node)-[:HAS_PART|HAS_CHAPTER|HAS_SECTION|HAS_ARTICLE|HAS_CLAUSE|HAS_POINT *0..]->(target)
           OR (target)-[:HAS_PART|HAS_CHAPTER|HAS_SECTION|HAS_ARTICLE|HAS_CLAUSE|HAS_POINT *0..]->(h_node)
        
        WITH DISTINCT target_uid, h_node
        MATCH (amending_doc:Document)-[*]->(amending_node)-[r:AMENDS]->(h_node)
        RETURN target_uid,
               amending_node.uid AS amending_uid,
               amending_doc.effect_date AS effect_date,
               labels(amending_node) AS amending_labels,
               amending_node.content AS amending_content,
               h_node.uid AS amended_uid,
               labels(h_node) AS amended_labels
        """
        result_map: dict[str, list[dict]] = {uid: [] for uid in uids}
        if not uids:
            return result_map
            
        with self._get_driver().session(database=self.database) as session:
            result = session.run(query, uids=uids)
            for record in result:
                target_uid = record["target_uid"]
                amending_labels = record["amending_labels"]
                amended_labels = record["amended_labels"]
                
                result_map[target_uid].append({
                    "amending_uid": record["amending_uid"],
                    "effect_date": record["effect_date"],
                    "amending_label": amending_labels[0] if amending_labels else "Unknown",
                    "amending_content": record["amending_content"],
                    "amended_uid": record["amended_uid"],
                    "amended_label": amended_labels[0] if amended_labels else "Unknown"
                })
        return result_map

    def multi_search(
        self, sub_queries: List[dict], k: int = 5
    ) -> dict[int, list["SearchResult"]]:
        """Search each sub-query independently against all node labels.

        Each sub-query is searched against ALL labels (Article, Clause, Point).
        Results are NOT merged or deduplicated — each sub-query keeps its own results.
        This preserves independence: the same node can appear in multiple sub-query results.

        Args:
            sub_queries: List of {"query": "..."} from decompose_query().
            k: Number of results per label per sub-query.

        Returns:
            Dict mapping sub-query index -> list of SearchResult.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_labels = ["Article", "Clause", "Point"]
        results: dict[int, list[SearchResult]] = {i: [] for i in range(len(sub_queries))}

        with ThreadPoolExecutor(max_workers=min(6, len(sub_queries))) as executor:
            futures = {
                executor.submit(self.search, all_labels, sq["query"], k): i
                for i, sq in enumerate(sub_queries)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return results


    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
