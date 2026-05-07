import os
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from legal_scraper.prompts import _QA_SYSTEM_PROMPT, _QA_USER_PROMPT

class AnswerGenerator:
    """Generates answers based on retrieved context or directly for conversational queries."""
    
    def __init__(self, local_model_url: str = None):
        self.local_model_url = local_model_url or os.getenv(
            "LOCAL_MODEL_URL", "https://vitalize-compacter-nephew.ngrok-free.dev/generate"
        )

    def _call_llm(self, prompt_text: str) -> str:
        @retry(
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=2, max=5),
            retry=retry_if_exception_type((requests.exceptions.RequestException, ValueError)),
            reraise=True,
        )
        def _do_request() -> str:
            payload = {
                "prompt": prompt_text,
                "max_new_tokens": 1024,
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
            
        return _do_request()

    def generate_rag_answer(self, query: str, context: str) -> str:
        """Generate answer using retrieved legal context."""
        try:
            prompt = f"<start_of_turn>user\n{_QA_SYSTEM_PROMPT}\n\n{_QA_USER_PROMPT.format(query=query, context=context)}<end_of_turn>\n<start_of_turn>model\n"
            return self._call_llm(prompt)
        except Exception as e:
            print(f"RAG Generation error: {e}")
            return "Xin lỗi, đã có lỗi xảy ra trong quá trình tổng hợp câu trả lời từ hệ thống."

    def generate_direct_answer(self, query: str) -> str:
        """Generate answer directly without context (for chitchat)."""
        try:
            sys_prompt = "Bạn là một Chatbot hỗ trợ tư vấn pháp luật giao thông đường bộ Việt Nam. Hãy trả lời câu hỏi của người dùng một cách thân thiện và ngắn gọn."
            prompt = f"<start_of_turn>user\n{sys_prompt}\n\nCâu hỏi: {query}<end_of_turn>\n<start_of_turn>model\n"
            return self._call_llm(prompt)
        except Exception as e:
            print(f"Direct Generation error: {e}")
            return "Xin lỗi, hiện tại tôi không thể xử lý câu hỏi này."
