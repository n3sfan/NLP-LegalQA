"""Query rewriter for multi-turn conversation using LangChain.

Uses ChatPromptTemplate + MessagesPlaceholder to contextualize follow-up
questions based on conversation history. Supports two LLM backends via
a single ``ChatOpenAI`` class (works with any OpenAI-compatible endpoint):

  "local"      – Your ngrok / vLLM / Ollama endpoint (default)
  "openrouter" – OpenRouter API

Environment variables:
    LLM_PROVIDER       : "local" (default) or "openrouter"

    # local (any OpenAI-compatible server, including ngrok)
    LLM_BASE_URL       : Base URL (default: http://localhost:8000/v1)
    LLM_MODEL          : Model name (default: gemma-4)
    LLM_API_KEY        : API key if required (default: not-needed)

    # openrouter
    OPENROUTER_API_KEY : API key (required)
    OPENROUTER_MODEL   : Model (default: google/gemma-4-26b-a4b-it:free)
"""

import os
from typing import List, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

from legal_scraper.prompts import _REWRITE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------

def create_chat_llm(
    provider: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 256,
    **kwargs,
) -> BaseChatModel:
    """Create a LangChain ChatOpenAI instance for the configured provider.

    Both providers use ``ChatOpenAI`` — the only difference is ``base_url``
    and credentials.  Switching is a one-line ``.env`` change.

    Providers:
        ``"local"`` (default)
            Any OpenAI-compatible server: ngrok endpoint, vLLM, Ollama, etc.
            Reads ``LLM_BASE_URL``, ``LLM_MODEL``, ``LLM_API_KEY``.

        ``"openrouter"``
            OpenRouter API.  Reads ``OPENROUTER_API_KEY``, ``OPENROUTER_MODEL``.

    Args:
        provider: ``"local"`` or ``"openrouter"``. Falls back to env
            ``LLM_PROVIDER`` (default ``"local"``).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        **kwargs: Extra keyword arguments forwarded to ``ChatOpenAI``.

    Returns:
        A ready-to-use LangChain chat model.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "local")

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY must be set when LLM_PROVIDER=openrouter"
            )
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free"),
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    else:  # local (ngrok, vLLM, Ollama — anything OpenAI-compatible)
        return ChatOpenAI(
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("LLM_API_KEY", "not-needed"),
            model=os.getenv("LLM_MODEL", "gemma-4"),
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Query Rewriter
# ---------------------------------------------------------------------------

class QueryRewriter:
    """Rewrites follow-up queries into standalone queries using chat history.

    Wraps a LangChain chain:  prompt → LLM → StrOutputParser.
    The prompt uses ``MessagesPlaceholder`` so chat history is injected
    natively as a list of ``HumanMessage`` / ``AIMessage``.

    Example::

        rewriter = QueryRewriter()
        history = [
            {"role": "user", "content": "vượt đèn đỏ xe máy phạt bao nhiêu"},
            {"role": "assistant", "content": "Theo Nghị định 100/2019..."},
        ]
        standalone = rewriter.rewrite(history, "nếu đi xe ô tô thì sao?")
        # => "mức phạt xe ô tô vượt đèn đỏ là bao nhiêu"
    """

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm or create_chat_llm(temperature=0.1, max_tokens=256)

        self.chain = (
            ChatPromptTemplate.from_messages([
                ("system", _REWRITE_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{query}"),
            ])
            | self.llm
            | StrOutputParser()
        )

    # -- helpers to convert dict history → LangChain messages --
    @staticmethod
    def _to_langchain_messages(
        chat_history: List[Dict[str, str]],
    ) -> List[HumanMessage | AIMessage]:
        """Convert ``[{"role": ..., "content": ...}]`` to LangChain messages."""
        messages = []
        for msg in chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        return messages

    def rewrite(
        self,
        chat_history: List[Dict[str, str]],
        query: str,
    ) -> str:
        """Rewrite *query* into a standalone query given *chat_history*.

        If there is no history, the original query is returned immediately
        (no LLM call — saves latency and tokens).

        Args:
            chat_history: List of ``{"role": "user"|"assistant", "content": "..."}``
                dicts representing previous turns.
            query: The latest user query (potentially a follow-up).

        Returns:
            A self-contained query string suitable for the retrieval pipeline.
        """
        if not chat_history:
            return query

        messages = self._to_langchain_messages(chat_history)

        try:
            rewritten = self.chain.invoke({
                "chat_history": messages,
                "query": query,
            })
            return rewritten.strip()
        except Exception as e:
            print(f"[!] Rewrite failed: {e}. Using original query.")
            return query
