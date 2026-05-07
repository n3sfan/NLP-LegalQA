import os
import json
import requests
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from legal_scraper.prompts import _DECOMPOSE_SYSTEM_PROMPT, _DECOMPOSE_USER_PROMPT

SubQuery = dict

class QueryDecomposer:
    """Handles parsing and decomposing user queries into subqueries using a local LLM API."""
    def __init__(self, local_model_url: str = None):
        self.local_model_url = local_model_url or os.getenv(
            "LOCAL_MODEL_URL", "https://vitalize-compacter-nephew.ngrok-free.dev/generate"
        )
        self.system_prompt = _DECOMPOSE_SYSTEM_PROMPT
        self.user_prompt = _DECOMPOSE_USER_PROMPT

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
        prompt_text = f"<start_of_turn>user\n{self.system_prompt}\n\n{self.user_prompt.format(query=query)}<end_of_turn>\n<start_of_turn>model\n"
        
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((requests.exceptions.RequestException, ValueError)),
            reraise=True,
        )
        def _call_llm() -> str:
            payload = {
                "prompt": prompt_text,
                "max_new_tokens": 512,
                "temperature": 0.1
            }
            headers = {
                "ngrok-skip-browser-warning": "true",
                "Content-Type": "application/json"
            }
            response = requests.post(self.local_model_url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            data = response.json()
            if "response" in data:
                return data["response"].strip()
            return ""

        raw_str = _call_llm()
        fallback = self._parse_json_fallback(raw_str)
        validated = [{"query": str(item["query"])} for item in fallback if isinstance(item, dict) and "query" in item]
        
        if not validated:
            raise ValueError(f"LLM generated invalid subqueries. Raw output: {raw_str}")
            
        return validated

class QueryRewriter:
    """Handles rewriting a simple user query into formal legal terminology."""
    def __init__(self, local_model_url: str = None):
        self.local_model_url = local_model_url or os.getenv(
            "LOCAL_MODEL_URL", "https://vitalize-compacter-nephew.ngrok-free.dev/generate"
        )
        from legal_scraper.prompts import _REWRITE_SYSTEM_PROMPT, _REWRITE_USER_PROMPT
        self.system_prompt = _REWRITE_SYSTEM_PROMPT
        self.user_prompt = _REWRITE_USER_PROMPT

    def _parse_json_fallback(self, text: str) -> List[dict]:
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

    def rewrite(self, query: str) -> List[SubQuery]:
        """Rewrite a single simple query."""
        prompt_text = f"<start_of_turn>user\n{self.system_prompt}\n\n{self.user_prompt.format(query=query)}<end_of_turn>\n<start_of_turn>model\n"
        
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((requests.exceptions.RequestException, ValueError)),
            reraise=True,
        )
        def _call_llm() -> str:
            payload = {
                "prompt": prompt_text,
                "max_new_tokens": 128,
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
            raw_str = _call_llm()
            fallback = self._parse_json_fallback(raw_str)
            validated = [{"query": str(item["query"])} for item in fallback if isinstance(item, dict) and "query" in item]
            
            if not validated:
                # If LLM failed to JSON format, just fallback to wrapping the raw string (assuming it output just the rewritten text)
                clean_str = raw_str.replace('"', '').replace('{', '').replace('}', '').replace('[', '').replace(']', '').replace('query:', '').strip()
                if clean_str:
                    return [{"query": clean_str}]
                raise ValueError(f"LLM generated invalid rewrite. Raw output: {raw_str}")
                
            return validated
        except Exception as e:
            print(f"Rewrite error: {e}. Falling back to original query.")
            return [{"query": query}]
