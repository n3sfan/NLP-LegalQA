import os
import json
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from legal_scraper.prompts import _DECOMPOSE_SYSTEM_PROMPT, _DECOMPOSE_USER_PROMPT
from legal_scraper.query_rewriter import create_chat_llm

SubQuery = dict

class QueryDecomposer:
    """Handles parsing and decomposing user queries into subqueries using LangChain."""
    def __init__(self, llm=None):
        self.llm = llm or create_chat_llm(temperature=0, max_tokens=512)
        self.chain = (
            ChatPromptTemplate.from_messages([
                ("system", _DECOMPOSE_SYSTEM_PROMPT),
                ("human", "{query}"),
            ])
            | self.llm
            | StrOutputParser()
        )

    def _parse_json_fallback(self, text: str) -> List[dict]:
        """Try to extract JSON array from LLM output, handling markdown code blocks."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                start = text.find("[")
                end = text.rfind("]")
                if start != -1 and end != -1 and end > start:
                    json_str = text[start : end + 1]
                    return json.loads(json_str)
            except Exception:
                pass
            return []

    def decompose(self, query: str) -> List[SubQuery]:
        """Decompose a user query into a list of SubQuery dictionaries."""
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=True,
        )
        def _call():
            return self.chain.invoke({"query": _DECOMPOSE_USER_PROMPT.format(query=query)})

        raw_str = _call()
        fallback = self._parse_json_fallback(raw_str)
        validated = [{"query": str(item["query"])} for item in fallback if isinstance(item, dict) and "query" in item]
        
        if not validated:
            raise ValueError(f"LLM generated invalid subqueries. Raw output: {raw_str}")
            
        return validated
