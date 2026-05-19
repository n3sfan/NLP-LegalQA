"""3-LLM majority vote classifier for legal QA relevance.

Supports three backends:
  - OllamaBackend    — local Ollama server (CPU/GPU, no setup needed)
  - LlamaCppBackend  — loads GGUF directly via llama-cpp-python (no server process)
  - VLLMBackend      — vLLM OpenAI-compatible server (GPU-accelerated, multi-model)

Usage:
    # Ollama (default)
    from llm.voter import LegalVoter, OllamaBackend
    backends = [OllamaBackend(model="qwen3:4b") for _ in range(3)]
    voter = LegalVoter(backends=backends, model_names=["qwen3:4b"]*3)

    # llama.cpp (loads GGUF directly — no server process needed)
    from llm.voter import LegalVoter, LlamaCppBackend
    backends = [
        LlamaCppBackend(model_path="/path/to/Qwen3-4B-Q4_K_M.gguf"),
        LlamaCppBackend(model_path="/path/to/nanbeige4.1-Q4_K_M.gguf"),
    ]
    voter = LegalVoter(backends=backends, model_names=["Qwen3-4B", "nanbeige4.1"])

    result = await voter.vote(question, law_text)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from os import environ
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class VoteResult:
    """Output of a majority vote run."""

    verdict: bool
    votes: list[bool] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    raw_responses: list[str] = field(default_factory=list)
    vote_durations_ms: list[float] = field(default_factory=list)  # per-model ms
    all_agreed: bool = False

    def __post_init__(self):
        if self.votes:
            self.all_agreed = all(v == self.votes[0] for v in self.votes)

    @property
    def total_duration_ms(self) -> float | None:
        return sum(self.vote_durations_ms) or None


# ---------------------------------------------------------------------------
# LLM Backend Protocol (swappable: Ollama, llama.cpp, vLLM, Gemini, …)
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMBackend(Protocol):
    """Minimal interface every LLM backend must implement."""

    async def ask(self, prompt: str) -> str:
        """Send a prompt and return the raw text response."""
        ...


# ---------------------------------------------------------------------------
# Ollama Backend
# ---------------------------------------------------------------------------


@dataclass
class OllamaBackend:
    """Calls a local Ollama server via langchain-ollama."""

    model: str = "llama3.2"
    base_url: str = "http://localhost:11434"
    timeout: float = 60.0

    async def ask(self, prompt: str) -> str:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            format="json",
            timeout=self.timeout,
        )
        raw = await asyncio.to_thread(llm.invoke, prompt)
        return raw.content


# ---------------------------------------------------------------------------
# llama.cpp Backend
# ---------------------------------------------------------------------------


@dataclass
class LlamaCppBackend:
    """Loads a GGUF model directly via llama-cpp-python (no server process).

    Each instance holds one model.  Create one backend per GGUF file.
    GPU offload is controlled by ``n_gpu_layers`` (set high to use GPU on
    systems with CUDA; set 0 to force CPU).

    Example (Colab with T4):
        LlamaCppBackend(
            model_path="/content/drive/MyDrive/HCMUS/NLP-LegalQA/models/Qwen3-4B-Q4_K_M.gguf",
            n_gpu_layers=33,   # T4: try 20-35 if OOM
            n_ctx=4096,
        )

    Requires: ``pip install llama-cpp-python``.
    """

    model_path: str
    n_gpu_layers: int = 0       # GPU layers (0 = CPU only, increase for GPU offload)
    n_ctx: int = 4096           # context window size
    temperature: float = 0.0     # deterministic for classification
    max_tokens: int = 256
    verbose: bool = False

    # Internal cache so the Llama model is only loaded once per path
    _llama: object = field(default=None, init=False, repr=False)

    def _get_llama(self):
        if self._llama is None:
            from llama_cpp import Llama

            self._llama = Llama(
                model_path=str(Path(self.model_path).expanduser().resolve()),
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=self.verbose,
            )
        return self._llama

    async def ask(self, prompt: str) -> str:
        llama = self._get_llama()
        # llama-cpp-python returns raw text; run in thread to keep async
        raw = await asyncio.to_thread(
            llama,
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        # Llama.__call__ returns a dict with "choices"[0]["text"]
        return raw["choices"][0]["text"].strip()


# ---------------------------------------------------------------------------
# vLLM Backend
# ---------------------------------------------------------------------------


@dataclass
class VLLMBackend:
    """Calls a vLLM OpenAI-compatible server via langchain-openai.

    vLLM serves models at e.g. http://localhost:8000/v1 with the OpenAI
    chat completions interface.

    After the server is up, set base_url to "http://localhost:8000/v1".
    """

    model: str = "Qwen/Qwen3-4B"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "vllm-secret-key"
    timeout: float | None = None  # None disables timeout for slow startup
    temperature: float = None
    max_tokens: int = None

    async def ask(self, prompt: str) -> str:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers={"Authorization": f"Bearer {self.api_key}"},
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout,
        )
        raw = await asyncio.to_thread(llm.invoke, prompt)

        if 'VOTER_TESTING' in environ:
            print(raw)
        return raw.content


# ---------------------------------------------------------------------------
# OpenRouter Backend
# ---------------------------------------------------------------------------


@dataclass
class OpenRouterBackend:
    """Calls OpenRouter via its OpenAI-compatible chat completions API."""

    model: str = "google/gemma-4-26b-a4b-it:free"
    api_key: str | None = None
    timeout: float | None = None  # None disables timeout for slow startup
    temperature: float = None
    max_tokens: int = None

    async def ask(self, prompt: str) -> str:
        from langchain_openai import ChatOpenAI

        api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY must be set for OpenRouter inference")

        llm = ChatOpenAI(
            model=self.model,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout,
        )
        raw = await asyncio.to_thread(llm.invoke, prompt)
        return raw.content


# ---------------------------------------------------------------------------
# Gemini Backend (future swap-in)
# ---------------------------------------------------------------------------


@dataclass
class GeminiBackend:
    """Calls Google Gemini API via google-genai SDK."""

    model: str = "gemini-2.0-flash-lite"
    api_key: str | None = None

    async def ask(self, prompt: str) -> str:
        from google.genai import genai

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": 0.0},
        )
        return response.text


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_vote_response(raw: str) -> bool | None:
    """Extract Yes/No from raw LLM text.

    Tries JSON first ({"answer": "Có"}), then keyword matching.
    Returns True for "Có"/"Yes", False for "Không"/"No", None if unclear.
    """
    raw = raw.strip()

    # 1. JSON {"answer": "Có" / "Không"}
    try:
        data = json.loads(raw)
        ans = data.get("answer", "").strip().lower()
        if ans in ("có", "yes", "co"):
            return True
        if ans in ("không", "no", "khong"):
            return False
    except json.JSONDecodeError:
        pass

    # 2. Keyword fallback
    raw_lower = raw.lower()
    if re.search(r"\b(có|yes|co)\b", raw_lower):
        return True
    if re.search(r"\b(không|no|khong)\b", raw_lower):
        return False

    logger.warning("Could not parse vote response: %r", raw[:200])
    return None


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def load_prompt_template() -> str:
    path = Path(__file__).parent / "prompt_classify.md"
    return path.read_text(encoding="utf-8")


def fill_prompt(template: str, question: str, law_text: str) -> str:
    return template.format(question=question, law_text=law_text)


# ---------------------------------------------------------------------------
# VoterAgent — one LLM call with its backend
# ---------------------------------------------------------------------------


@dataclass
class VoterAgent:
    """Wraps a single LLM backend and knows how to classify one vote."""

    backend: LLMBackend
    model_name: str = "unknown"

    async def vote(self, question: str, law_text: str) -> tuple[bool, str, float]:
        """Returns (vote, raw_response, elapsed_ms)."""
        import time
        template = load_prompt_template()
        prompt = template.format(question=question, law_text=law_text)
        # print('Vote prompt:', prompt)
        max_retries = 3
        raw = ""
        elapsed_ms = 0
        
        for attempt in range(max_retries):
            t0 = time.monotonic()
            try:
                raw = await self.backend.ask(prompt)
                elapsed_ms = (time.monotonic() - t0) * 1000
                if raw and raw.strip():
                    break
                else:
                    logger.warning(
                        "[%s] empty response from backend (attempt %d/%d) — elapsed_ms=%.1f",
                        self.model_name, attempt+1, max_retries, elapsed_ms
                    )
            except Exception as exc:
                elapsed_ms = (time.monotonic() - t0) * 1000
                logger.error(
                    "[%s] backend.ask() failed (attempt %d/%d) — type=%s msg=%s",
                    self.model_name, attempt+1, max_retries, type(exc).__name__, exc
                )
                raw = f"ERROR: {exc}"
            
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
        parsed = parse_vote_response(raw)
        if parsed is None:
            parsed = False  # ambiguous → vote No
        
        if 'VOTER_TESTING' in environ:
            print(f'parsed: {parsed}  [{elapsed_ms:.0f}ms]  [{self.model_name}]')
        return parsed, raw, elapsed_ms


# ---------------------------------------------------------------------------
# LegalVoter — runs N voters concurrently, takes majority
# ---------------------------------------------------------------------------


@dataclass
class LegalVoter:
    """Runs multiple VoterAgent instances concurrently and returns majority verdict."""

    backends: list[LLMBackend]
    model_names: list[str] | None = None

    def __post_init__(self):
        if self.model_names is None:
            self.model_names = [b.model if hasattr(b, "model") else str(b) for b in self.backends]
        self.agents = [
            VoterAgent(backend=b, model_name=n)
            for b, n in zip(self.backends, self.model_names)
        ]

    async def vote(self, question: str, law_text: str) -> VoteResult:
        tasks = [agent.vote(question, law_text) for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        votes: list[bool] = []
        models: list[str] = []
        raw_responses: list[str] = []
        vote_durations_ms: list[float] = []

        for agent, result in zip(self.agents, results):
            if isinstance(result, Exception):
                logger.error("Vote failed for %s: %s", agent.model_name, result)
                continue
            vote, raw, elapsed_ms = result
            votes.append(vote)
            models.append(agent.model_name)
            raw_responses.append(raw)
            vote_durations_ms.append(elapsed_ms)

        if not votes:
            raise RuntimeError("All votes failed")

        verdict = sum(votes) >= len(votes) // 2
        print('votes cnt:', sum(votes), len(votes))
        return VoteResult(
            verdict=verdict,
            votes=votes,
            models=models,
            raw_responses=raw_responses,
            vote_durations_ms=vote_durations_ms,
        )


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------


async def majority_vote(
    question: str,
    law_text: str,
    models: list[str] | None = None,
    backend: str = "llama_cpp",
    base_url: str | None = None,
    api_key: str = "vllm-secret-key",
    n_gpu_layers: int = 0,
    n_ctx: int = 4096,
) -> VoteResult:
    """Run one voter per model and return a majority verdict.

    Args:
        question:    The legal question.
        law_text:    The law text / retrieved passage to classify.
        models:      List of model identifiers or GGUF paths (depends on backend).
        backend:     "ollama" | "llama_cpp" | "vllm"
        base_url:    Server base URL for ollama/vllm.  Defaults to localhost:11434
                     (ollama) or localhost:8000/v1 (vllm).
        api_key:     API key for vllm (ignored for others).
        n_gpu_layers: GPU layers for llama_cpp (0 = CPU).
        n_ctx:       Context window size for llama_cpp.
    """
    if models is None:
        models = ["qwen3:4b", "qwen3:4b", "qwen3:4b"]

    backends: list[LLMBackend] = []

    if backend == "llama_cpp":
        for m in models:
            backends.append(
                LlamaCppBackend(
                    model_path=m,  # path to GGUF file
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=n_ctx,
                    temperature=0.0,
                )
            )
        model_names = [Path(m).stem for m in models]  # friendly name from filename
        voter = LegalVoter(backends=backends, model_names=model_names)

    elif backend == "vllm":
        _base_url = base_url or "http://localhost:8000/v1"
        for m in models:
            backends.append(VLLMBackend(model=m, base_url=_base_url, api_key=api_key))
        voter = LegalVoter(backends=backends, model_names=models)

    else:  # ollama
        _base_url = base_url or "http://localhost:11434"
        for m in models:
            backends.append(OllamaBackend(model=m, base_url=_base_url))
        voter = LegalVoter(backends=backends, model_names=models)

    return await voter.vote(question, law_text)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    async def _smoke_test():
        logging.basicConfig(level=logging.INFO)

        # Unit tests that don't need a live server
        print("--- parse_vote_response unit tests ---")
        assert parse_vote_response('{"answer": "Có"}') is True, "Có JSON"
        assert parse_vote_response('{"answer": "Không"}') is False, "Không JSON"
        assert parse_vote_response("Có") is True, "Có plain"
        assert parse_vote_response("Không") is False, "Không plain"
        assert parse_vote_response("Yes") is True, "Yes"
        assert parse_vote_response("No") is False, "No"
        assert parse_vote_response("dung roi, Khong") is False, "Khong in text"
        assert parse_vote_response(" totally random ") is None, "unclear"
        print("All parse tests passed.")

        print("--- prompt template load ---")
        tpl = load_prompt_template()
        assert "{question}" in tpl and "{law_text}" in tpl, "template vars"
        print(f"Template loaded ({len(tpl)} chars).")

        filled = fill_prompt(tpl, "Lái xe uống rượu phạt bao nhiêu?", "Điều 6. Phạt 20 triệu")
        assert "Lái xe uống rượu phạt bao nhiêu?" in filled
        assert "Điều 6. Phạt 20 triệu" in filled
        print("fill_prompt OK.")

        q = (
            "Điều 6. Phạt tiền từ 18.000.000 đồng đến 20.000.000 đồng "
            "đối với người điều khiển xe thực hiện hành vi điều khiển xe "
            "trên đường mà trong máu có nồng độ cồn vượt quá 50 miligam."
        )
        question = "Lái xe có nồng độ cồn vượt quá mức quy định bị phạt bao nhiêu?"

        print("\n--- integration test (llama_cpp) ---")
        print("[SKIPPED — requires GGUF file path. To run locally:")
        print("  uv run python -m src.llm.voter")

        # Uncomment to run with a real GGUF:
        # try:
        #     result = await majority_vote(
        #         question, q, backend="llama_cpp",
        #         models=["/path/to/model.Q4_K_M.gguf"],
        #         n_gpu_layers=33, n_ctx=4096,
        #     )
        #     print(f"llama_cpp result: {result}")
        # except Exception as e:
        #     print(f"[SKIPPED] llama_cpp error: {e}")

        print("\n--- integration test (Ollama) ---")
        try:
            result = await majority_vote(
                question, q, backend="ollama", models=["llama3.2"]
            )
            print(f"Ollama result: {result}")
        except RuntimeError as e:
            print(f"[SKIPPED] Ollama not running: {e}")

        print("\n--- integration test (vLLM) ---")
        try:
            result = await majority_vote(
                question, q, backend="vllm",
                models=["Qwen/Qwen3-4B"],
                base_url="http://localhost:8000/v1",
                api_key="vllm-secret-key",
            )
            print(f"vLLM result: {result}")
        except Exception as e:
            print(f"[SKIPPED] vLLM not running: {e}")

    asyncio.run(_smoke_test())
