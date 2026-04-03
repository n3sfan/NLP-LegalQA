#!/usr/bin/env python3
"""
Legal QA Dataset Generator using Gemini API
Generates reasoning dataset for legal QA with rate limiting (RPM, TPM)

Usage:
    export GEMINI_API_KEY="your-api-key"
    python gemini_dataset_generator.py --input data2/input.json

Input JSON format:
[
    {
        "document_name": "Nghị định 168/2024/NĐ-CP",
        "content": "Full text content...",
        "aspects_list": ["aspect1", "aspect2", ...]
    }
]
"""

import json
import os
import time
import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import google.genai as genai
from google.genai import types


# ─── Rate Limit Config ───────────────────────────────────────────────────────

@dataclass
class RateLimitConfig:
    """Rate limit configuration for Gemini API"""
    rpm: int = 15            # Requests per minute (Gemini 3.1 Flash Lite default)
    tpm: int = 1_000_000     # Tokens per minute (1M for Flash Lite)
    max_retries: int = 3


# ─── Rate Limiter ─────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Token-bucket style rate limiter tracking RPM and TPM.
    Sleeps before allowing the next request when either limit is hit.
    """

    def __init__(self, config: RateLimitConfig):
        self.rpm = config.rpm
        self.tpm = config.tpm
        # [(timestamp, token_count), ...] — entries older than 60s are pruned
        self._token_log: list[tuple[float, int]] = []
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int):
        """Block until both RPM and TPM limits allow the next request."""
        async with self._lock:
            now = time.time()
            self._prune(now)

            # 1) RPM check — max N requests in the trailing 60-s window
            while self._request_count(now) >= self.rpm:
                sleep_s = 60.0 - (now - self._oldest_request_ts(now))
                print(f"  [RateLimit] RPM ({self.rpm}/min) — sleeping {sleep_s:.1f}s")
                await asyncio.sleep(sleep_s)
                now = time.time()
                self._prune(now)

            # 2) TPM check — max N tokens in the trailing 60-s window
            while self._token_sum(now) + estimated_tokens > self.tpm:
                sleep_s = 60.0 - (now - self._oldest_token_ts(now))
                print(f"  [RateLimit] TPM ({self.tpm:,}/min) — sleeping {sleep_s:.1f}s")
                await asyncio.sleep(sleep_s)
                now = time.time()
                self._prune(now)

            # Record this request
            self._token_log.append((now, estimated_tokens))

    # ── private helpers ───────────────────────────────────────────────────────

    def _prune(self, now: float):
        self._token_log = [(ts, t) for ts, t in self._token_log if now - ts < 60]

    def _request_count(self, now: float) -> int:
        return sum(1 for ts, _ in self._token_log if now - ts < 60)

    def _token_sum(self, now: float) -> int:
        return sum(t for ts, t in self._token_log if now - ts < 60)

    def _oldest_request_ts(self, now: float) -> float:
        return min(ts for ts, _ in self._token_log if now - ts < 60)

    def _oldest_token_ts(self, now: float) -> float:
        return min(ts for ts, _ in self._token_log if now - ts < 60)


# ─── Input / Output Models ────────────────────────────────────────────────────

@dataclass
class InputItem:
    document_name: str
    content: str
    aspects_list: list[str]


@dataclass
class DatasetEntry:
    """Single QA entry (shared across tasks)"""
    question: str
    choices: list[str]
    answer: int
    reasoning: str
    legal_document: Optional[str] = None
    specific_question: Optional[str] = None
    document_name: Optional[str] = None


# ─── Main Generator ───────────────────────────────────────────────────────────

class GeminiDatasetGenerator:
    """Generate legal QA dataset (task1 + task2) using Gemini API."""

    # Default prompt template (inline — can be overridden via file)
    DEFAULT_PROMPT = (
        "Bạn là chuyên gia pháp lý Việt Nam. Dựa trên văn bản pháp luật được cung cấp, "
        "hãy thực hiện các nhiệm vụ sau một cách chính xác và chi tiết.\n\n"
        "## Văn bản pháp luật:\n"
        "{doc_content}\n\n"
        "## Danh sách khía cạnh cần phân tích:\n"
        "{aspects_list}\n\n"
        "## Nhiệm vụ 1: Đánh giá khả năng trả lời câu hỏi pháp luật\n"
        "Với mỗi câu hỏi pháp luật cụ thể và đoạn trích dẫn điều luật, hãy đánh giá xem "
        "đoạn trích dẫn đó có đủ thông tin để trả lời câu hỏi hay không.\n"
        "- Định dạng: dict JSON với trường legal_document, specific_question, question, "
        "choices (list), answer (int), reasoning\n\n"
        "## Nhiệm vụ 2: Câu hỏi trắc nghiệm pháp luật\n"
        "Với mỗi khía cạnh, tạo câu hỏi trắc nghiệm pháp luật với 4 lựa chọn (đáp án đúng random).\n"
        "- Định dạng: dict JSON với trường question, choices (list), answer (int), reasoning\n\n"
        "## Output Format\n"
        'Trả về JSON với 2 khóa: "task1", "task2", mỗi khóa chứa mảng ví dụ.\n'
        'Ví dụ: {{"task1": [{{...}}], "task2": [{{...}}]}}'
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash-lite",
        rate_config: Optional[RateLimitConfig] = None,
        output_dir: str = "data2/output",
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set. Pass api_key or set env variable.")

        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        self.rate_config = rate_config or RateLimitConfig()
        self.rate_limiter = RateLimiter(self.rate_config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ───────────────────────────────────────────────────────────

    def generate(
        self,
        input_items: list[InputItem],
        prompt_file: Optional[str] = None,
        progress_file: Optional[str] = None,
        max_items: Optional[int] = None,
    ) -> dict:
        """
        Synchronous entry point.  Loads prompt, iterates items, returns merged dict.
        Each item produces { "task1": [...], "task2": [...] }.
        Only task1 and task2 are included (task3 / open-ended is excluded per spec).
        """
        template = self._load_prompt(prompt_file)
        processed = self._load_progress(progress_file)

        merged: dict = {"task1": [], "task2": []}
        items = input_items[:max_items] if max_items else input_items

        for idx, item in enumerate(items, 1):
            doc_id = self._doc_id(item.document_name)

            if doc_id in processed:
                print(f"[{idx}/{len(items)}] SKIP (done): {item.document_name}")
                continue

            print(f"[{idx}/{len(items)}] GENERATE: {item.document_name}")

            prompt = self._build_prompt(template, item)
            est_tokens = self._estimate_tokens(prompt)

            print(f"{prompt = }")

            # ── rate-limit + call ───────────────────────────────────────────
            asyncio.run(self._rate_limiter_sync(est_tokens))
            result = self._call_gemini(prompt)
            # ─────────────────────────────────────────────────────────────────

            for key in ["task1", "task2"]:
                if key in result:
                    for entry in result[key]:
                        entry["document_name"] = item.document_name
                    merged[key].extend(result[key])

            processed.add(doc_id)
            self._save_progress(processed, progress_file)
            self._save_intermediate(merged, idx)

            # Brief pause between requests to avoid hammering
            time.sleep(0.5)

        self._save_final(merged)
        return merged

    # ── private helpers ───────────────────────────────────────────────────────

    async def _rate_limiter_sync(self, tokens: int):
        await self.rate_limiter.acquire(tokens)

    def _load_prompt(self, path: Optional[str]) -> str:
        if path:
            return Path(path).read_text(encoding="utf-8")
        return self.DEFAULT_PROMPT

    def _build_prompt(self, template: str, item: InputItem) -> str:
        aspects = "\n".join(f"- {a}" for a in item.aspects_list)
        # Use .replace() instead of str.format() so curly braces in document
        # content are not interpreted as placeholders
        # Smart/curly quotes used in prompt.txt: both are U+201D (")
        smart_q = "\u201d"
        return (
            template
            .replace("{document_name}", item.document_name)
            .replace(f'{{doc_info[{smart_q}content{smart_q}]}}', item.content)
            .replace("{aspects_list}", aspects)
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~1 token per 4 chars for Vietnamese."""
        return len(text) // 4

    @staticmethod
    def _doc_id(name: str) -> str:
        return name.replace(" ", "_")[:60]

    def _call_gemini(self, prompt: str) -> dict:
        """Make one API call with retries."""
        for attempt in range(1, self.rate_config.max_retries + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        topP=0.95,
                        maxOutputTokens=8192,
                    ),
                )
                return self._parse_response(resp.text)

            except Exception as exc:  # broad catch — network, quota, etc.
                print(f"  [Attempt {attempt}] Error: {exc}")
                if attempt < self.rate_config.max_retries:
                    wait = 2 ** attempt
                    print(f"  Retrying in {wait}s ...")
                    time.sleep(wait)
                else:
                    print("  [FATAL] Max retries reached, returning empty.")
                    return {"task1": [], "task2": []}
        return {"task1": [], "task2": []}  # unreachable

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """Strip markdown fences and parse JSON."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return json.loads(text)

    # ── progress tracking ─────────────────────────────────────────────────────

    @staticmethod
    def _load_progress(path: Optional[str]) -> set:
        if path and Path(path).exists():
            return set(json.loads(Path(path).read_text(encoding="utf-8")))
        return set()

    @staticmethod
    def _save_progress(processed: set, path: Optional[str]):
        if path:
            Path(path).write_text(
                json.dumps(list(processed), ensure_ascii=False),
                encoding="utf-8",
            )

    # ── file output ───────────────────────────────────────────────────────────

    def _save_intermediate(self, merged: dict, idx: int):
        path = self.output_dir / f"intermediate_{idx}.json"
        path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_final(self, merged: dict):
        path = self.output_dir / "final_dataset.json"
        path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        task1, task2 = len(merged["task1"]), len(merged["task2"])
        print(f"\n✓ Done — task1: {task1} | task2: {task2} | saved to {path}")


# ─── Input loader ────────────────────────────────────────────────────────────

def _resolve_content(raw: str) -> str:
    """If raw looks like a file path and the file exists, read it; else return raw."""
    path = Path(raw.strip())
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8")
    return raw


def load_input_items(path: str) -> list[InputItem]:
    """Load documents from JSON. Supports list or dict-of-dicts formats.

    The "content" field may be:
      - Actual text content
      - A path to a .txt file (will be read and replaced)
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    items: list[InputItem] = []
    if isinstance(data, list):
        for entry in data:
            items.append(InputItem(
                document_name=entry.get("document_name", "Unknown"),
                content=_resolve_content(
                    entry.get("content", entry.get("noi_dung", ""))
                ),
                aspects_list=entry.get("aspects_list", entry.get("aspects", [])),
            ))
    elif isinstance(data, dict):
        for key, entry in data.items():
            items.append(InputItem(
                document_name=entry.get("document_name", key),
                content=_resolve_content(
                    entry.get("content", entry.get("noi_dung", ""))
                ),
                aspects_list=entry.get("aspects_list", entry.get("aspects", [])),
            ))
    return items


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate legal QA reasoning dataset via Gemini API"
    )
    parser.add_argument(
        "--input", "-i", default="data2/input.json",
        help="Input JSON file with documents"
    )
    parser.add_argument(
        "--prompt", "-p", default="data2/prompt.txt",
        help="Prompt template file"
    )
    parser.add_argument(
        "--output", "-o", default="data2/output",
        help="Output directory"
    )
    parser.add_argument(
        "--progress", default="data2/progress.json",
        help="Progress tracking file"
    )
    parser.add_argument(
        "--max", "-m", type=int, default=None,
        help="Max items to process (default: all)"
    )
    parser.add_argument(
        "--rpm", type=int, default=15,
        help="Requests per minute limit (default: 15)"
    )
    parser.add_argument(
        "--tpm", type=int, default=1_000_000,
        help="Tokens per minute limit (default: 1,000,000)"
    )
    parser.add_argument(
        "--model", "-M", default="gemini-3.1-flash-lite-preview",
        help="Gemini model name (default: gemini-3.1-flash-lite)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY environment variable:")
        print('  export GEMINI_API_KEY="your-key"')
        exit(1)

    print(f"Loading input from: {args.input}")
    input_items = load_input_items(args.input)
    print(f"Loaded {len(input_items)} document(s)")

    generator = GeminiDatasetGenerator(
        api_key=api_key,
        model=args.model,
        rate_config=RateLimitConfig(rpm=args.rpm, tpm=args.tpm),
        output_dir=args.output,
    )

    merged = generator.generate(
        input_items=input_items,
        prompt_file=args.prompt,
        progress_file=args.progress,
        max_items=args.max,
    )

    print(f"\nResults: task1={len(merged['task1'])} | task2={len(merged['task2'])}")
