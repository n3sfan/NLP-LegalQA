import os
import json
from typing import Literal
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from legal_scraper.prompts import _ROUTER_SYSTEM_PROMPT, _ROUTER_USER_PROMPT
from legal_scraper.query_rewriter import create_chat_llm

IntentType = Literal["direct_answer", "retrieve", "reject"]

class QueryRouter:
    """Classifies user queries to determine the appropriate response strategy."""
    
    def __init__(self, llm=None):
        self.llm = llm or create_chat_llm(temperature=0.1, max_tokens=64)
        self.chain = (
            ChatPromptTemplate.from_messages([
                ("system", _ROUTER_SYSTEM_PROMPT),
                ("human", "{query}"),
            ])
            | self.llm
            | StrOutputParser()
        )

    def route(self, query: str) -> IntentType:
        """Classify the user query. Fallback to 'retrieve' on failure."""
        @retry(
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=2, max=5),
            reraise=True,
        )
        def _call():
            return self.chain.invoke({"query": _ROUTER_USER_PROMPT.format(query=query)})

        try:
            raw_response = _call()
            return self._parse_intent(raw_response)
        except Exception as e:
            print(f"Router error: {e}. Falling back to 'retrieve'.")
            return "retrieve"

    def _parse_intent(self, response_text: str) -> IntentType:
        """Parse the JSON response to extract the intent."""
        text = response_text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            
        try:
            data = json.loads(text)
            intent = data.get("intent", "retrieve").lower()
            if intent in ["direct_answer", "retrieve", "reject"]:
                return intent
        except json.JSONDecodeError:
            # Simple fallback parsing using string matching
            if '"intent": "direct_answer"' in text or "'intent': 'direct_answer'" in text:
                return "direct_answer"
            elif '"intent": "reject"' in text or "'intent': 'reject'" in text:
                return "reject"
        
        return "retrieve"
