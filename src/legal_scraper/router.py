import os
import json
import requests
from typing import Literal
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from legal_scraper.prompts import _ROUTER_SYSTEM_PROMPT, _ROUTER_USER_PROMPT

IntentType = Literal["direct_answer", "retrieve", "reject"]

class QueryRouter:
    """Classifies user queries to determine the appropriate response strategy."""
    
    def __init__(self, local_model_url: str = None):
        self.local_model_url = local_model_url or os.getenv(
            "LOCAL_MODEL_URL", "https://vitalize-compacter-nephew.ngrok-free.dev/generate"
        )
        self.system_prompt = _ROUTER_SYSTEM_PROMPT
        self.user_prompt = _ROUTER_USER_PROMPT

    def route(self, query: str) -> IntentType:
        """Classify the user query. Fallback to 'retrieve' on failure."""
        prompt_text = f"<start_of_turn>user\n{self.system_prompt}\n\n{self.user_prompt.format(query=query)}<end_of_turn>\n<start_of_turn>model\n"
        
        @retry(
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=2, max=5),
            retry=retry_if_exception_type((requests.exceptions.RequestException, ValueError)),
            reraise=True,
        )
        def _call_llm() -> str:
            payload = {
                "prompt": prompt_text,
                "max_new_tokens": 64,
                "temperature": 0.1
            }
            headers = {
                "ngrok-skip-browser-warning": "true",
                "Content-Type": "application/json"
            }
            response = requests.post(self.local_model_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            if "response" in data:
                return data["response"].strip()
            return ""

        try:
            raw_response = _call_llm()
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
