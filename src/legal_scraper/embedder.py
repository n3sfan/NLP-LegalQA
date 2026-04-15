from collections import namedtuple
from typing import List

from neo4j import GraphDatabase
from pyvi.ViTokenizer import tokenize

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
        tokenized = [tokenize(text) for text in texts]
        return self._embeddings.embed_documents(tokenized)

    def embed_query(self, text: str) -> List[float]:
        """Tokenize the query and embed via the underlying model."""
        return self._embeddings.embed_query(tokenize(text))


class Neo4jEmbedder:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
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
                model_kwargs={"device": "cuda"},
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
        Returns a dict keyed by (uid, label) → {"content": str, "title": str|None}.
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

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
