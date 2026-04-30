from collections import namedtuple
import os
from dataclasses import dataclass
from typing import List, Optional
import json

from neo4j import GraphDatabase
from pyvi.ViTokenizer import tokenize

SearchResult = namedtuple("SearchResult", ["uid", "label", "score"])

_DECOMPOSE_SYSTEM_PROMPT = """Bạn là chuyên gia tiền xử lý truy vấn pháp lý cho hệ thống RAG.

Nhiệm vụ:
Phân tích câu hỏi của người dùng thành các sub-query ngắn, độc lập, chuẩn hóa theo thuật ngữ pháp luật Việt Nam, để phục vụ tìm kiếm văn bản pháp luật.

Nguyên tắc bắt buộc:
1. Chỉ tách khi thật sự có nhiều vấn đề pháp lý khác nhau.
   - Nếu câu hỏi chỉ chứa 1 hành vi, 1 yêu cầu, hoặc 1 vấn đề pháp lý duy nhất, KHÔNG tách nhỏ.
   - Khi đó, chỉ trả về 1 sub-query đã được chuẩn hóa.

2. Tách đúng một vấn đề pháp lý cho mỗi sub-query.
   - Mỗi hành vi vi phạm, mỗi nghĩa vụ, mỗi quyền, mỗi yêu cầu xử lý pháp lý phải là một sub-query riêng.
   - Không gộp nhiều lỗi hoặc nhiều căn cứ pháp lý khác nhau trong cùng một sub-query.

3. Giữ đủ ngữ cảnh cần thiết.
   - Mỗi sub-query phải tự đủ nghĩa khi đứng một mình.
   - Phải giữ các thông tin quan trọng như: chủ thể, phương tiện, hành vi, hậu quả, quan hệ pháp lý.
   - Nếu phương tiện được suy ra rõ từ ngữ cảnh, hãy nêu rõ theo cách pháp lý phù hợp.

4. Chuẩn hóa sang thuật ngữ pháp luật Việt Nam.
   - Dùng ngôn ngữ pháp lý hoặc hành chính tương ứng.
   - Chuyển từ ngữ đời thường sang cách diễn đạt chính thức.
   - Ví dụ:
     - "nhậu xỉn", "say xỉn" -> "điều khiển phương tiện mà trong máu hoặc hơi thở có nồng độ cồn"
     - "lấn tuyến" -> "đi không đúng phần đường, làn đường"
     - "tông xe làm hư xe người ta" -> "gây tai nạn giao thông làm hư hỏng tài sản của người khác"

5. Bảo toàn mục đích tra cứu.
   - Nếu câu hỏi gốc hỏi về mức phạt, phải giữ các từ khóa như "mức xử phạt", "xử phạt vi phạm hành chính".
   - Nếu hỏi về bồi thường, phải giữ "trách nhiệm bồi thường thiệt hại".
   - Nếu hỏi về hình sự, phải giữ "truy cứu trách nhiệm hình sự" hoặc cụm tương đương phù hợp.

6. Viết sub-query dưới dạng cụm từ tìm kiếm, không viết thành câu hỏi.
   - Không dùng từ nghi vấn như: ai, gì, thế nào, bao nhiêu, không hay chỉ.
   - Ưu tiên cụm danh từ hoặc cụm pháp lý ngắn, rõ ràng.

7. Chỉ dùng dữ kiện có thật trong câu gốc.
   - Không tự suy diễn thêm lỗi, hậu quả, hay căn cứ pháp lý không được nêu ra.
   - Không tự mở rộng sang hành vi khác nếu người dùng không nhắc đến.

8. Số lượng:
   - Tối thiểu 1 sub-query, tối đa 6 sub-queries.
   - Nếu chỉ có một vấn đề pháp lý duy nhất, trả về đúng 1 sub-query.
   - Loại bỏ các sub-query trùng ý hoặc quá gần nhau.

Quy tắc đầu ra:
- Chỉ trả về JSON array hợp lệ.
- Mỗi phần tử có đúng một khóa: "query".
- Không bọc trong markdown.
- Không giải thích.
- Không thêm bất kỳ văn bản nào khác.

Định dạng bắt buộc:
[
  {"query": "sub-query 1"},
  {"query": "sub-query 2"}
]
Ví dụ mẫu:
User: "Chạy xe máy vượt đèn đỏ tông hư rào nhà người ta thì đi tù không hay chỉ đền tiền?"
Output:
[
  {"query": "Mức xử phạt vi phạm hành chính hành vi điều khiển xe mô tô, xe gắn máy không chấp hành hiệu lệnh đèn tín hiệu giao thông"},
  {"query": "Trách nhiệm bồi thường thiệt hại dân sự do tai nạn giao thông gây thiệt hại tài sản của người khác"},
  {"query": "Căn cứ truy cứu trách nhiệm hình sự đối với hành vi vi phạm quy định về tham gia giao thông đường bộ gây thiệt hại tài sản"}
]
"""

_DECOMPOSE_USER_PROMPT = """Câu hỏi cần phân tích:
{query}

Yêu cầu:
- Trả về đúng một JSON array.
- Không dùng markdown, không dùng code fence.
- Không giải thích, không bình luận, không thêm chữ ngoài JSON.
- Nếu câu hỏi chỉ có một vấn đề pháp lý duy nhất, chỉ trả về 1 phần tử đã chuẩn hóa.
- Output phải bắt đầu bằng [ và kết thúc bằng ].
"""
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
        local_model_url: Optional[str] = None,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = None
        self._embedding_model = None
        self.local_model_url = local_model_url or os.getenv(
            "LOCAL_MODEL_URL", "https://vitalize-compacter-nephew.ngrok-free.dev/generate"
        )

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

        Calls local Gemma model to split the original question.
        Returns DecomposeResult with sub-queries, raw response, and success flag.
        Useful for inspecting how the model decomposes the query.

        Args:
            query: Original Vietnamese question.

        Returns:
            DecomposeResult(sub_queries, raw_response, success).
        """
        import requests
        from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

        prompt_text = f"<start_of_turn>user\n{_DECOMPOSE_SYSTEM_PROMPT}\n\n{_DECOMPOSE_USER_PROMPT.format(query=query)}<end_of_turn>\n<start_of_turn>model\n"

        @retry(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=10, min=10, max=120),
            reraise=True,
        )
        def _call_llm():
            data = {
                "prompt": prompt_text,
                "max_new_tokens": 512,
                "temperature": 0.1
            }
            headers = {
                "ngrok-skip-browser-warning": "true",
                "Content-Type": "application/json"
            }
            response = requests.post(self.local_model_url, json=data, headers=headers)
            response.raise_for_status()
            return response.json()["response"]

        try:
            raw_str = _call_llm()
            print(f"--- RAW LLM OUTPUT ---\n{raw_str}\n----------------------") # Add this line
            fallback = _parse_json_fallback(raw_str)
            validated = [{"query": str(item["query"])} for item in fallback if isinstance(item, dict) and "query" in item]
            return DecomposeResult(
                sub_queries=validated,
                reasoning=raw_str,
                success=len(validated) > 0,
            )
        except Exception as e:
            return DecomposeResult(sub_queries=[], reasoning=str(e), success=False)

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
