from collections import namedtuple
from dataclasses import dataclass
from typing import List, Optional
import json

from neo4j import GraphDatabase
from pyvi.ViTokenizer import tokenize

SearchResult = namedtuple("SearchResult", ["uid", "label", "score"])


_DECOMPOSE_SYSTEM_PROMPT = """Bạn là hệ thống tiền xử lý truy vấn pháp lý.
Nhiệm vụ của bạn là phân tích câu hỏi phức tạp của người dùng thành các câu hỏi con (sub-queries) đơn giản và hoàn toàn độc lập, yêu cầu là phải sử dụng các thuật ngữ pháp luật giống với hệ thống pháp luật việt nam.

QUY TẮC TÁCH CÂU HỎI:
1. Tách bạch các hành vi, đối tượng hoặc vấn đề khác nhau thành các sub-queries riêng biệt. 
   (VD: "Đi máy vượt đèn đỏ và quên mang bằng lái thì phạt sao?" -> Tách thành 2 truy vấn riêng cho 2 lỗi).
2. TRUYỀN NGỮ CẢNH: Mỗi sub-query phải tự chứa đầy đủ ngữ cảnh của câu gốc. 
   (VD: Câu gốc nhắc đến "xe máy", thì chữ "xe máy" phải xuất hiện trong mọi sub-query để không bị mất bối cảnh khi tìm kiếm độc lập).
3. Tối thiểu 1 sub-query, tối đa 6 sub-queries.
4. Chỉ tách câu hỏi, TUYỆT ĐỐI không tự suy diễn câu trả lời.
5. Nếu câu hỏi dùng từ ngữ đời thường (VD: lấn tuyến, vượt đèn đỏ), hãy chuẩn hóa nó sang thuật ngữ pháp lý tương đương (VD: đi không đúng phần đường, không chấp hành hiệu lệnh đèn tín hiệu) trong sub-query.
6. Biến đổi query gốc dạng câu hỏi thành dạng không câu hỏi (declarative statement) trong sub-query để tăng khả năng tìm kiếm chính xác của vector search.
   (VD: "Đi máy vượt đèn đỏ thì bị phạt thế nào?" -> "Không chấp hành hiệu lệnh đèn tín hiệu.").
Trả lời bằng định dạng JSON array chính xác như sau:
```json
[
  {{"query": "nội dung câu hỏi con 1 đã có đủ ngữ cảnh"}},
  {{"query": "nội dung câu hỏi con 2 đã có đủ ngữ cảnh"}}
]
```"""

_DECOMPOSE_USER_PROMPT = """Câu hỏi cần phân tích:
{query}

Hãy suy nghĩ từng bước (chain-of-thought), sau đó trả lời JSON array."""


def _parse_json_fallback(text: str) -> List[dict]:
    """Try to extract JSON array from LLM output, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except json.JSONDecodeError:
        # Try extracting first JSON-like block
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return []


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


SubQuery = dict  # {"label": str, "text": str}


@dataclass
class DecomposeResult:
    """Result of decompose_query_debug(). Holds sub-queries + CoT reasoning."""

    sub_queries: List[SubQuery]
    reasoning: str  # full raw LLM output (CoT + JSON)
    success: bool  # True if JSON parsed successfully


class Neo4jEmbedder:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        openrouter_api_key: Optional[str] = None,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = None
        self._embedding_model = None
        self._openrouter_api_key = openrouter_api_key
        self._llm_model: Optional["ChatOpenAI"] = None

    def _get_llm_model(self) -> "ChatOpenAI":
        if self._llm_model is None:
            from langchain_openai import ChatOpenAI  # type: ignore[attr-defined]

            self._llm_model = ChatOpenAI(
                model="google/gemma-4-26b-a4b-it:free",
                api_key=self._openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                timeout=60,
            )
        return self._llm_model

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

    def fetch_amends(self, uids: list[str]) -> dict[str, list[dict]]:
        """Fetch amends for the given node UIDs.
        
        An item is considered amended if it is amended directly or if any of its parent nodes
        (Article, Chapter, etc.) are amended.
        Returns a mapping from target_uid to a list of amendment dictionaries.
        """
        query = """
        UNWIND $uids AS target_uid
        MATCH path = (d:Document)-[:HAS_PART|HAS_CHAPTER|HAS_SECTION|HAS_ARTICLE|HAS_CLAUSE|HAS_POINT *]->(target)
        WHERE target.uid = target_uid
        WITH target_uid, nodes(path) AS hierarchy_nodes
        UNWIND hierarchy_nodes AS h_node
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
        self, sub_queries: List[SubQuery], k: int = 5
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

    def decompose_query(self, query: str) -> List[SubQuery]:
        """Alias for decompose_query_debug(). Returns only sub-queries list.

        Args:
            query: Original Vietnamese question.

        Returns:
            List of {"label": "Article"|"Clause"|"Point", "text": str}.
            Returns empty list if LLM output cannot be parsed as JSON.
        """
        result = self.decompose_query_debug(query)
        return result.sub_queries

    def decompose_query_debug(self, query: str) -> DecomposeResult:
        """Decompose a complex legal question into independent sub-queries (debug mode).

        Calls OpenRouter (gemma-4-26b-a4b-it:free) to split the original question.
        Returns DecomposeResult with sub-queries, full CoT reasoning, and success flag.
        Useful for inspecting how the model decomposes the query.

        Args:
            query: Original Vietnamese question.

        Returns:
            DecomposeResult(sub_queries, reasoning, success).
        """
        if self._openrouter_api_key is None:
            raise RuntimeError(
                "openrouter_api_key not set. "
                "Pass openrouter_api_key when initializing Neo4jEmbedder."
            )

        from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_core.output_parsers import JsonOutputParser

        llm = self._get_llm_model()
        parser = JsonOutputParser()
        messages = [
            SystemMessage(content=_DECOMPOSE_SYSTEM_PROMPT),
            HumanMessage(content=_DECOMPOSE_USER_PROMPT.format(query=query)),
        ]

        @retry(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=10, min=10, max=120),
            reraise=True,
        )
        def _call_llm():
            return llm.invoke(messages)

        raw = _call_llm()
        raw_str = raw.content if hasattr(raw, "content") else str(raw)

        try:
            parsed = parser.invoke(raw)
            if not isinstance(parsed, list):
                return DecomposeResult(sub_queries=[], reasoning=raw_str, success=False)
            validated: List[SubQuery] = [
                {"query": str(item["query"])}
                for item in parsed
                if isinstance(item, dict) and "query" in item
            ]
            return DecomposeResult(sub_queries=validated, reasoning=raw_str, success=True)
        except Exception:
            fallback = _parse_json_fallback(raw_str)
            validated = [{"query": str(item["query"])} for item in fallback if isinstance(item, dict) and "query" in item]
            return DecomposeResult(
                sub_queries=validated,
                reasoning=raw_str,
                success=len(validated) > 0,
            )

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
