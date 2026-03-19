"""Amend extraction using NuExtract API."""

from __future__ import annotations

import json
import requests
from dataclasses import asdict
from typing import Any

from legal_scraper.models import Amend


# NuExtract API Configuration
NUEXTRACT_API_KEY = "f26935ec75924b19a80881d11d331119"
NUEXTRACT_PROJECT_ID = "bab70407-6c8c-4c5c-971a-00b7844e8c15"
NUEXTRACT_BASE_URL = "https://nuextract.ai/api"


# JSON Schema for amend extraction
AMEND_SCHEMA = {
    "type": "object",
    "properties": {
        "amends": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "amending_doc_identity": {
                        "type": "string",
                        "description": "Mã văn bản luật sửa đổi (ví dụ: 67/2020/QH14)"
                    },
                    "amending_article": {
                        "type": "string",
                        "description": "Điều trong văn bản sửa đổi (ví dụ: Điều 1)"
                    },
                    "amending_clause": {
                        "type": "string",
                        "description": "Khoản trong điều sửa đổi (ví dụ: khoản 1)"
                    },
                    "amend_type": {
                        "type": "string",
                        "enum": ["sửa đổi", "bổ sung", "bãi bỏ", "thay thế"],
                        "description": "Loại sửa đổi"
                    },
                    "target_doc_identity": {
                        "type": "string",
                        "description": "Mã văn bản bị sửa đổi (ví dụ: 41/2004/QH11)"
                    },
                    "target_article": {
                        "type": "string",
                        "description": "Điều bị sửa đổi (ví dụ: Điều 163)"
                    },
                    "target_clause": {
                        "type": "string",
                        "description": "Khoản bị sửa đổi (ví dụ: khoản 1)"
                    },
                    "target_point": {
                        "type": "string",
                        "description": "Điểm bị sửa đổi (ví dụ: điểm đ)"
                    },
                    "original_text": {
                        "type": "string",
                        "description": "Nội dung gốc trước khi sửa đổi (nếu có)"
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Nội dung mới được thêm/sửa đổi"
                    }
                },
                "required": ["amending_article", "amend_type"]
            }
        }
    }
}


FEW_SHOT_EXAMPLES = [
    {
        "input": "Điều 1. Sửa đổi, bổ sung một số điều của Luật Chứng khoán\n1. Bổ sung khoản 49 vào sau khoản 48 Điều 4 như sau:\n\"49. Thao túng thị trường chứng khoán là việc thực hiện một trong các hành vi sau đây:\na) Sử dụng một hoặc nhiều tài khoản giao dịch của mình hoặc của người khác hoặc thông đồng liên tục mua, bán chứng khoán nhằm tạo ra cung, cầu giả tạo;\nb) Đặt lệnh mua và bán cùng loại chứng khoán trong cùng ngày giao dịch...\"",
        "output": {
            "amending_article": "1",
            "amending_clause": "1",
            "amend_type": "bổ sung",
            "target_doc_identity": "54/2019/QH14",
            "target_article": "4",
            "target_clause": "49",
            "target_point": None
        }
    },
    {
        "input": "Điều 1. Sửa đổi, bổ sung một số điều của Luật Chứng khoán\n2. Sửa đổi, bổ sung điểm d khoản 1 Điều 9 như sau:\n\"d) Quản lý, thanh tra, kiểm tra, giám sát hoạt động nghiệp vụ chứng khoán của Sở giao dịch chứng khoán Việt Nam và công ty con, Tổng công ty lưu ký và bù trừ chứng khoán Việt Nam và công ty con; chấp thuận các quy chế nghiệp vụ của Sở giao dịch chứng khoán Việt Nam...\"",
        "output": {
            "amending_article": "1",
            "amending_clause": "2",
            "amend_type": "sửa đổi, bổ sung",
            "target_doc_identity": "54/2019/QH14",
            "target_article": "9",
            "target_clause": "1",
            "target_point": "d"
        }
    },
    {
        "input": "3. Bổ sung một số điểm, khoản của Điều 11 như sau:\na) Bổ sung điểm e vào sau điểm đ khoản 1 như sau:\n\"e) Nhà đầu tư nước ngoài là cá nhân có quốc tịch nước ngoài, tổ chức thành lập theo pháp luật nước ngoài thực hiện hoạt động đầu tư kinh doanh tại Việt Nam.\"",
        "output": {
            "amending_article": "1",
            "amending_clause": "3a",
            "amend_type": "bổ sung",
            "target_doc_identity": "54/2019/QH14",
            "target_article": "11",
            "target_clause": "1",
            "target_point": "e"
        }
    },
    {
        "input": "4. Bổ sung Điều 11a vào sau Điều 11 như sau:\n\"Điều 11a. Trách nhiệm của tổ chức, cá nhân liên quan đến hồ sơ, tài liệu báo cáo\n1. Tổ chức, cá nhân tham gia vào quá trình lập hồ sơ, tài liệu báo cáo liên quan đến hoạt động về chứng khoán và thị trường chứng khoán phải chịu trách nhiệm trước pháp luật về tính hợp pháp, chính xác, trung thực và đầy đủ của hồ sơ, tài liệu báo cáo...\"",
        "output": {
            "amending_article": "1",
            "amending_clause": "4",
            "amend_type": "bổ sung",
            "target_doc_identity": "54/2019/QH14",
            "target_article": "11a",
            "target_clause": None,
            "target_point": None
        }
    },
    {
        "input": "5. Sửa đổi, bổ sung khoản 3 Điều 12 như sau:\n\"3. Thực hiện hành vi thao túng thị trường chứng khoán.\"",
        "output": {
            "amending_article": "1",
            "amending_clause": "5",
            "amend_type": "sửa đổi, bổ sung",
            "target_doc_identity": "54/2019/QH14",
            "target_article": "12",
            "target_clause": "3",
            "target_point": None
        }
    }
]


def parse_preamble(preamble: str) -> dict[str, str]:
    """
    Parse the preamble to extract mapping of doc_identity -> law name.
    
    Input: "...Luật Chứng khoán số 54/2019/QH14, Luật Kế toán số 88/2015/QH13..."
    Output: {"54/2019/QH14": "Luật Chứng khoán", "88/2015/QH13": "Luật Kế toán"}

    Returns:
        Dict mapping doc_identity (e.g., "54/2019/QH14") -> tên văn bản
    """
    import re

    result = {}

    # Pattern: "Luật X số YY/YYYY/QHZZ" hoặc "Luật số YY/YYYY/QHZZ"
    # Tìm các cụm: "Luật [tên] số [mã]" hoặc "Luật số [mã]"
    pattern = r'Luật\s+([A-Za-zÀ-ỹ\s]+?)\s+số\s*(\d{1,3}/\d{4}/QH\d{1,2})'
    matches = re.findall(pattern, preamble)

    for law_name, doc_id in matches:
        result[doc_id] = law_name.strip()

    if not result:
        pattern2 = r'số\s*(\d{1,3}/\d{4}/QH\d{1,2})'
        matches2 = re.findall(pattern2, preamble)
        for doc_id in matches2:
            result[doc_id] = doc_id

    return result


def map_target_doc_identity(amends: list[dict], preamble_map: dict[str, str]) -> list[dict]:
    """
    Map target_doc_identity dựa trên target_article và preamble_map.

    Args:
        amends: List of amend dictionaries
        preamble_map: Dict mapping doc_identity -> law name

    Returns:
        List of amends với target_doc_identity đã được map
    """
    import re

    doc_id_pattern = re.compile(r'^\d{1,3}/\d{4}/QH\d{1,2}$')

    # Lấy danh sách doc_identities theo thứ tự
    doc_ids = list(preamble_map.keys())

    # Debug
    with open("debug_map.txt", "w", encoding="utf-8") as f:
        f.write(f"preamble_map: {preamble_map}\n")
        f.write(f"doc_ids: {doc_ids}\n")
        f.write(f"num amends: {len(amends)}\n")

    for amend in amends:
        target_doc = amend.get("target_doc_identity", "")


        if not doc_id_pattern.match(target_doc or ""):
            # assign the first doc_identity from preamble_map if target_doc_identity is missing or invalid
            if doc_ids:
                amend["target_doc_identity"] = doc_ids[0]

    return amends


class AmendExtractor:
    """Extract amendment relationships from legal documents using NuExtract API."""

    def __init__(self, api_key: str | None = None,
                 project_id: str | None = None):
        self.api_key = api_key if api_key else NUEXTRACT_API_KEY
        self.project_id = project_id if project_id else NUEXTRACT_PROJECT_ID
        self.base_url = NUEXTRACT_BASE_URL

    def extract(self, content: str, doc_identity: str) -> list[dict]:
        """
        Extract amendment relationships from document content.

        Args:
            content: The article/clause content to analyze
            doc_identity: The document identity (e.g., "67/2020/QH14")

        Returns:
            List of amend dictionaries
        """
        url = f"{self.base_url}/projects/{self.project_id}/extract"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Build prompt with few-shot examples
        examples_text = "\n\n".join([
            f"Input: {ex['input']}\nOutput: {json.dumps(ex['output'], ensure_ascii=False)}"
            for ex in FEW_SHOT_EXAMPLES
        ])

        prompt = f"""Văn bản sửa đổi: {doc_identity}

Nội dung cần phân tích:
{content}

## Ví dụ:
{examples_text}

## Yêu cầu:
Trích xuất thông tin sửa đổi, bổ sung theo định dạng JSON như các ví dụ trên."""

        payload = {
            "content": prompt
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json()

            # Parse the response to extract amends
            return self._parse_response(result, doc_identity)

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return []

    def _parse_response(self, response: dict, doc_identity: str) -> list[dict]:
        """Parse NuExtract API response into Amend objects."""
        amends = []

        try:
            # Extract data from response
            data = response

            # Check for different response formats
            if isinstance(data, dict):
                if "data" in data:
                    data = data["data"]
                if "result" in data:
                    data = data["result"]
                if "output" in data:
                    data = data["output"]
                if "amends" in data:
                    data = data["amends"]
                elif isinstance(data, str):
                    data = json.loads(data)

            # Handle array response
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        if "amends" in item:
                            amends.extend(self._extract_amends_from_item(item["amends"], doc_identity))
                        else:
                            amend = self._create_amend_dict(item, doc_identity)
                            if amend:
                                amends.append(amend)
            elif isinstance(data, dict):
                if "amends" in data:
                    amends.extend(self._extract_amends_from_item(data["amends"], doc_identity))
                else:
                    amend = self._create_amend_dict(data, doc_identity)
                    if amend:
                        amends.append(amend)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse response: {e}")
            print(f"Response: {response}")

        return amends

    def _extract_amends_from_item(self, item: Any, doc_identity: str) -> list[dict]:
        """Extract amends from a single item."""
        amends = []

        if isinstance(item, list):
            for sub_item in item:
                if isinstance(sub_item, dict):
                    amend = self._create_amend_dict(sub_item, doc_identity)
                    if amend:
                        amends.append(amend)
        elif isinstance(item, dict):
            amend = self._create_amend_dict(item, doc_identity)
            if amend:
                amends.append(amend)

        return amends

    def _create_amend_dict(self, item: dict, doc_identity: str) -> dict | None:
        """Create an amend dictionary from parsed item."""
        if not item:
            return None

        return {
            "amending_doc_identity": item.get("amending_doc_identity", doc_identity),
            "amending_article": item.get("amending_article", ""),
            "amending_clause": item.get("amending_clause"),
            "amend_type": item.get("amend_type", "sửa đổi"),
            "target_doc_identity": item.get("target_doc_identity"),
            "target_article": item.get("target_article"),
            "target_clause": item.get("target_clause"),
            "target_point": item.get("target_point"),
            "original_text": item.get("original_text"),
            "new_text": item.get("new_text")
        }

    def extract_from_articles(self, articles: list[dict], doc_identity: str, clauses: list[dict] | None = None, points: list[dict] | None = None, preamble: str = "") -> list[dict]:
        """
        Process amendments per article - groups clauses/points by article.

        Args:
            articles: List of article dictionaries with 'number', 'content', 'title'
            clauses: List of clause dictionaries with 'number', 'content', 'parent_article'
            points: List of point dictionaries with 'letter', 'content', 'parent_article', 'parent_clause'
            doc_identity: Document identity
            preamble: Document preamble (for parsing to get target_doc_identity)

        Returns:
            List of all extracted amends
        """
        # Parse preamble to get mapping of doc_identity -> law name
        preamble_map = {}
        if preamble:
            preamble_map = parse_preamble(preamble)

        all_amends = []

        # Group clauses by parent_article
        clauses_by_article = {}
        if clauses:
            for clause in clauses:
                article_num = clause.get("parent_article", "")
                if article_num not in clauses_by_article:
                    clauses_by_article[article_num] = []
                clauses_by_article[article_num].append(clause)

        # Group points by parent_article
        points_by_article = {}
        if points:
            for point in points:
                article_num = point.get("parent_article", "")
                if article_num not in points_by_article:
                    points_by_article[article_num] = []
                points_by_article[article_num].append(point)

        # Build preamble context for prompt
        preamble_context = ""
        if preamble_map:
            preamble_context = "Văn bản gốc được sửa đổi: " + ", ".join([
                f"{name} ({doc_id})" for doc_id, name in preamble_map.items()
            ])

        # Process each article
        for article in articles:
            article_num = article.get("number", "")
            article_title = article.get("title", "")

            # Skip if no clauses or points for this article
            article_clauses = clauses_by_article.get(article_num, [])
            article_points = points_by_article.get(article_num, [])

            if not article_clauses and not article_points:
                continue

            # Build content from clauses
            clauses_text = ""
            if article_clauses:
                clauses_lines = []
                for clause in article_clauses:
                    clause_num = clause.get("number", "")
                    content = clause.get("content", "")
                    clauses_lines.append(f"{clause_num}. {content}")
                clauses_text = "\n".join(clauses_lines)

            # Build content from points
            points_text = ""
            if article_points:
                points_lines = []
                for point in article_points:
                    letter = point.get("letter", "")
                    content = point.get("content", "")
                    clause_ref = point.get("parent_clause", "")
                    points_lines.append(f"Điểm {letter}, Khoản {clause_ref}: {content}")
                points_text = "\n".join(points_lines)

            # Build full prompt for this article
            full_text = f"""Văn bản sửa đổi: {doc_identity}
{preamble_context}

Điều {article_num}. {article_title}

Các khoản:
{clauses_text}

Các điểm:
{points_text}

Yêu cầu: Trích xuất tất cả thông tin sửa đổi, bổ sung từ nội dung trên."""

            # Extract amendments for this article
            amends = self.extract(full_text, doc_identity)

            # Post-process to ensure correct fields
            for amend in amends:
                if not amend.get("amending_article"):
                    amend["amending_article"] = article_num
                if not amend.get("amending_doc_identity"):
                    amend["amending_doc_identity"] = doc_identity

            all_amends.extend(amends)

        # Map target_doc_identity using preamble_map
        if preamble_map:
            all_amends = map_target_doc_identity(all_amends, preamble_map)

        return all_amends

    def _contains_amend_keywords(self, content: str, title: str) -> bool:
        """Check if content contains amendment keywords."""
        text = f"{content} {title}".lower()
        keywords = ["sửa đổi", "bổ sung", "bãi bỏ", "thay thế", "điều"]
        return any(kw in text for kw in keywords)


def extract_amends(content: str, doc_identity: str) -> list[dict]:
    """Convenience function to extract amendments."""
    extractor = AmendExtractor()
    return extractor.extract(content, doc_identity)
