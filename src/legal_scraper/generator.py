import os
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from legal_scraper.prompts import _QA_SYSTEM_PROMPT, _QA_USER_PROMPT
from legal_scraper.query_rewriter import create_chat_llm

class AnswerGenerator:
    """Generates answers based on retrieved context or directly for conversational queries."""
    
    def __init__(self, llm=None):
        self.llm = llm or create_chat_llm(temperature=0.1, max_tokens=4096)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        chain = (
            ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            | self.llm
            | StrOutputParser()
        )

        @retry(
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=2, max=5),
            reraise=True,
        )
        def _do_request() -> str:
            return chain.invoke({"input": user_prompt})
            
        return _do_request()

    def generate_rag_answer(self, query: str, context: str, current_date: str | None = None, rewritten_query: str | None = None) -> str:
        """Generate answer using retrieved legal context."""
        from datetime import datetime
        date_str = current_date or datetime.now().strftime("%Y-%m-%d")
        # Build optional rewritten query section
        if rewritten_query and rewritten_query != query:
            rewritten_section = f"\n[Câu hỏi đã được làm rõ (dùng để tra cứu)]:\n{rewritten_query}"
        else:
            rewritten_section = ""
        try:
            user_prompt = _QA_USER_PROMPT.format(query=query, context=context, current_date=date_str, rewritten_section=rewritten_section)
            return self._call_llm(_QA_SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            print(f"RAG Generation error: {e}")
            return "Xin lỗi, đã có lỗi xảy ra trong quá trình tổng hợp câu trả lời từ hệ thống."

    def generate_direct_answer(self, query: str) -> str:
        """Generate answer directly without context (for chitchat)."""
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        try:
            sys_prompt = f"Bạn là một Chatbot hỗ trợ tư vấn pháp luật giao thông đường bộ Việt Nam. Hãy trả lời câu hỏi của người dùng một cách thân thiện và ngắn gọn. Ngày hiện tại: {date_str}."
            return self._call_llm(sys_prompt, f"Câu hỏi: {query}")
        except Exception as e:
            print(f"Direct Generation error: {e}")
            return "Xin lỗi, hiện tại tôi không thể xử lý câu hỏi này."
