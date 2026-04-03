"""
Legal QA Dataset Generator using NotebookLM
Generates reasoning dataset for legal QA using Google NotebookLM's chat API.

Prerequisites:
    conda activate ml-env
    notebooklm login          # opens browser — sign in with Google account

Usage:
    python notebooklm_dataset_generator.py --input data2/input.json

Input JSON format (same as gemini version):
[
    {
        "document_name": "Nghị định 168/2024/NĐ-CP",
        "content": "Full text content...",
        "aspects_list": ["aspect1", "aspect2", ...]
    }
]

Output:
    data2/output/notebooklm_final_dataset.json
        {"task1": [...], "task2": [...]}
"""

import json
import os
import time
import argparse
import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# NotebookLM SDK
from notebooklm import NotebookLMClient

# ─── Config ────────────────────────────────────────────────────────────────────

@dataclass
class RateLimitConfig:
    """Rate limit configuration (NotebookLM uses HTTP 429 / retry-after)"""
    max_retries: int = 5
    base_delay: float = 2.0          # seconds — exponential backoff base
    source_wait_timeout: float = 120.0  # seconds — wait for source processing
    generation_timeout: float = 300.0  # seconds — wait for quiz/generation


# ─── Input / Output Models ─────────────────────────────────────────────────────

@dataclass
class InputItem:
    document_name: str
    content: str
    aspects_list: list[str]


# ─── NotebookLM Client Wrapper ─────────────────────────────────────────────────

class NotebookLMGenerator:
    """
    Generates legal QA dataset (task1 + task2) using NotebookLM.

    Design decisions:
    - ONE notebook per run (reused across documents)
      → Each document is added as a TEXT source, processed, then removed.
      → Keeps the Google account tidy; avoids per-document notebook clutter.
    - task1 (Yes/No reasoning) + task2 (4-choice MCQ) are requested in a
      single comprehensive question; response is parsed for both.
    - If the response is not parseable as JSON, falls back to asking each
      task type separately.
    - Progress tracked by document name (same skip-on-restart behaviour).
    """

    def __init__(
        self,
        rate_config: Optional[RateLimitConfig] = None,
        output_dir: str = "data2/output",
        notebook_title: str = "LegalQA-Dataset",
        client: NotebookLMClient = None 
    ):
        self.rate_config = rate_config or RateLimitConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.notebook_title = notebook_title
        self._client: Optional[NotebookLMClient] = None
        self._notebook_id: Optional[str] = None
        self._source_ids: list[str] = []
        self._client = client
        self._prompt_template = (Path(__file__).parent.parent / "data2" / "prompt.txt").read_text()

    # ── public async API ───────────────────────────────────────────────────────

    async def generate(
        self,
        input_items: list[InputItem],
        progress_file: Optional[str] = None,
        max_items: Optional[int] = None,
    ) -> dict:
        """
        Main entry point.  Returns {"task1": [...], "task2": [...]}.
        """
        # Create or reuse notebook
        await self._ensure_notebook()

        processed = self._load_progress(progress_file)
        merged: dict = {"task1": [], "task2": []}
        items = input_items[:max_items] if max_items else input_items

        for idx, item in enumerate(items, 1):
            doc_id = self._doc_id(item.document_name)

            if doc_id in processed:
                print(f"[{idx}/{len(items)}] SKIP (done): {item.document_name}")
                continue

            print(f"[{idx}/{len(items)}] PROCESS: {item.document_name}")

            # 1. Add document as source
            source_id = await self._add_source(item)

            # 2. Ask task1 + task2
            result = await self._generate_tasks(item, source_id, self._prompt_template)

            # 3. Tag entries with document name
            for key in ["task1", "task2"]:
                if key in result:
                    for entry in result[key]:
                        entry["document_name"] = item.document_name
                    merged[key].extend(result[key])

            # 4. Remove source (keep notebook clean)
            # if source_id:
            #     await self._remove_source(source_id)

            processed.add(doc_id)
            self._save_progress(processed, progress_file)
            self._save_intermediate(merged, idx)

            # Brief pause between documents
            await asyncio.sleep(2)

        await self._close()
        self._save_final(merged)
        return merged

    # ── notebook lifecycle ───────────────────────────────────────────────────

    async def _ensure_notebook(self):
        """Create the working notebook if it doesn't exist."""
        notebooks = await self._client.notebooks.list()
        for nb in notebooks:
            if nb.title == self.notebook_title:
                self._notebook_id = nb.id
                print(f"  [Notebook] Reusing existing: {nb.id}")
                return

        nb = await self._client.notebooks.create(self.notebook_title)
        self._notebook_id = nb.id
        print(f"  [Notebook] Created new: {nb.id}")

    async def _add_source(self, item: InputItem) -> Optional[str]:
        """Add document content as a text source. Returns source_id or None on failure."""
        # Truncate content if too long (NotebookLM has a limit ~200K chars)
        content = item.content[:180_000]
        try:
            # Find the source id we just added
            sources = await self._client.sources.list(self._notebook_id)
            for src in sources:
                print(src.title, item.document_name)
                if src.title == item.document_name:
                    sid = src.id
                    self._source_ids.append(sid)
                    print(f"  [Source] Already Added: {item.document_name} ({sid})")
                    return sid
                
            await self._client.sources.add_text(
                self._notebook_id,
                title=item.document_name,
                content=item.content,
                wait=True,
                # timeout=self.rate_config.source_wait_timeout,
            )
            

            print(f"  [Source] Added but ID not found — using first source")
            if sources:
                sid = sources[0].id
                self._source_ids.append(sid)
                return sid
            return None

        except Exception as exc:
            print(f"  [Source] Error adding {item.document_name}: {exc}")
            return None

    async def _remove_source(self, source_id: str):
        """Remove a source from the notebook."""
        try:
            await self._client.sources.delete(self._notebook_id, source_id)
            if source_id in self._source_ids:
                self._source_ids.remove(source_id)
            print(f"  [Source] Removed: {source_id}")
        except Exception as exc:
            print(f"  [Source] Error removing {source_id}: {exc}")

    async def _close(self):
        """Close the client (no-op: NotebookLMClient has no close method)."""
        pass

    # ── generation ───────────────────────────────────────────────────────────

    async def _generate_tasks(
        self, item: InputItem, source_id: Optional[str], prompt_template: str
    ) -> dict:
        """
        Ask NotebookLM for task1 + task2 using the exact prompt template,
        with {document_name} and {aspects_list} replaced.
        """
        aspects_str = (
            "\n".join(f"- {a}" for a in item.aspects_list)
            if item.aspects_list
            else "(toàn bộ văn bản)"
        )

        question = prompt_template.format(
            document_name=item.document_name,
            doc_info_content=item.content,
            aspects_list=aspects_str,
        )

        result = await self._ask_with_retry(question, source_id)
        parsed = self._parse_response(result)

        if parsed and "task1" in parsed and "task2" in parsed:
            return parsed

        # Fallback: ask each task separately
        print("  [Fallback] Failed to parse — asking tasks separately")

        task1_result = await self._ask_with_retry(
            f"Văn bản: {item.document_name}\n\n"
            "Hãy tạo câu hỏi Yes/No (JSON, key: task1):",
            source_id,
        )
        task2_result = await self._ask_with_retry(
            f"Văn bản: {item.document_name}\nCác khía cạnh:\n{aspects_str}\n\n"
            "Hãy tạo câu hỏi tự luận pháp luật (JSON, key: task2):",
            source_id,
        )

        task1 = self._parse_response(task1_result) or {}
        task2 = self._parse_response(task2_result) or {}
        return {"task1": task1.get("task1", []), "task2": task2.get("task2", [])}

    async def _ask_with_retry(
        self, question: str, source_id: Optional[str]
    ) -> Optional[str]:
        """Ask NotebookLM with exponential-backoff retry on 429 / network errors."""
        for attempt in range(1, self.rate_config.max_retries + 1):
            try:
                result = await self._client.chat.ask(
                    self._notebook_id,
                    question,
                    # source_ids=[source_id] if source_id else None,
                )
                print('answer', result.answer)
                return result.answer

            except Exception as exc:
                msg = str(exc).lower()
                if "429" in msg or "rate limit" in msg or "throttl" in msg:
                    delay = self.rate_config.base_delay * (2 ** (attempt - 1))
                    print(f"  [429/RateLimit] attempt {attempt}, sleeping {delay:.1f}s")
                    await asyncio.sleep(delay)
                else:
                    print(f"  [Error] attempt {attempt}: {exc}")
                    if attempt < self.rate_config.max_retries:
                        delay = self.rate_config.base_delay * (2 ** (attempt - 1))
                        await asyncio.sleep(delay)
                    else:
                        return None

        return None

    # ── response parsing ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(raw: Optional[str]) -> Optional[dict]:
        """
        Extract JSON from a NotebookLM answer string.
        Handles:
          - Markdown ```json ... ``` fences
          - Text before/after JSON block
          - Partial/truncated JSON (tries to recover)
        """
        if not raw:
            return None

        text = raw.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strip markdown fences
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```", "", text)
        text = text.strip()

        # Strip notebook cite
        text = re.sub(r"\[(\d+\,\s)*\d+\]\.", ".", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract first {...} block
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        print(f"  [ParseError] Could not extract JSON from response (len={len(raw)})")
        return None

    # ── progress / file I/O ───────────────────────────────────────────────────

    @staticmethod
    def _doc_id(name: str) -> str:
        return name.replace(" ", "_")[:60]

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

    def _save_intermediate(self, merged: dict, idx: int):
        path = self.output_dir / f"notebooklm_intermediate_{idx}.json"
        path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [Saved] {path}")

    def _save_final(self, merged: dict):
        path = self.output_dir / "notebooklm_final_dataset.json"
        path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✓ Done — task1: {len(merged['task1'])} | task2: {len(merged['task2'])}")
        print(f"  → {path}")


# ─── Input loader (same as gemini version) ────────────────────────────────────

def _resolve_content(raw: str) -> str:
    """If raw is an existing file path, read and return file contents."""
    path = Path(raw.strip())
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8")
    return raw


def load_input_items(path: str) -> list[InputItem]:
    """Load input JSON. Supports list-of-dicts or dict-of-dicts."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items: list[InputItem] = []

    def make_item(key: str, entry: dict) -> InputItem:
        return InputItem(
            document_name=entry.get("document_name", key),
            content=_resolve_content(entry.get("content", entry.get("noi_dung", ""))),
            aspects_list=entry.get("aspects_list", entry.get("aspects", [])),
        )

    if isinstance(data, list):
        for entry in data:
            items.append(make_item(entry.get("document_name", "Unknown"), entry))
    elif isinstance(data, dict):
        for key, entry in data.items():
            items.append(make_item(key, entry))

    return items


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Legal QA dataset generator via NotebookLM")
    sub = p.add_subparsers(dest="command", help="Subcommands")

    gen = sub.add_parser("generate", help="Batch generate from input JSON")
    gen.add_argument("--input", "-i", default="data2/input.json")
    gen.add_argument("--output", "-o", default="data2/output")
    gen.add_argument("--progress", default="data2/notebooklm_progress.json")
    gen.add_argument("--max", "-m", type=int, default=None)
    gen.add_argument("--notebook", default="LegalQA-Dataset")
    gen.add_argument("--retries", type=int, default=5)
    gen.add_argument("--source-timeout", type=float, default=120.0)

    sam = sub.add_parser("sample", help="Interactive: one prompt → flat per-QA JSON files → repeat")
    sam.add_argument("--input", "-i", default="data2/input.json")
    sam.add_argument("--output", "-o", default="data2/samples")
    sam.add_argument("--notebook", default="LegalQA-Dataset")

    return p.parse_args()


async def main():
    args = parse_args()

    if args.command == "sample":
        await _run_sample(args)
    else:
        await _run_generate(args)


async def _run_generate(args):
    print("=" * 60)
    print("NotebookLM Legal QA Dataset Generator")
    print("=" * 60)
    print(f"Input  : {args.input}")
    print(f"Output : {args.output}")
    print()

    async with await NotebookLMClient.from_storage() as client:
        input_items = load_input_items(args.input)
        print(f"Loaded {len(input_items)} document(s)\n")

        generator = NotebookLMGenerator(
            output_dir=args.output,
            notebook_title=args.notebook,
            rate_config=RateLimitConfig(
                max_retries=args.retries,
                source_wait_timeout=args.source_timeout,
            ),
            client=client
        )

        merged = await generator.generate(
            input_items=input_items,
            progress_file=args.progress,
            max_items=args.max,
        )

        print(f"\nResults: task1={len(merged['task1'])} | task2={len(merged['task2'])}")


async def _run_sample(args):
    """
    Interactive loop: one NotebookLM call per input document.
    - Saves one RAW file (prompt + raw response + parsed JSON).
    - ALSO saves ONE flat JSON per extracted QA entry (task1 / task2).
    - After each document the user chooses: r=regenerate, n=next, s=skip, q=quit.
    """
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # raw logs go in a sub-folder
    raw_dir = output_dir / "_raw"
    raw_dir.mkdir(exist_ok=True)

    prompt_path = Path(__file__).parent.parent / "data2" / "prompt.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8")

    input_items = load_input_items(args.input)
    if not input_items:
        print("No items found in data2/input.json")
        return

    print("=" * 60)
    print("NotebookLM Interactive Sample Generator")
    print("=" * 60)
    print(f"Documents : {len(input_items)}")
    print(f"Output dir: {output_dir}")
    print(f"Raw logs  : {raw_dir}")
    print(f"Prompt    : {prompt_path}")
    print()

    # global sample counter for flat filenames
    global_idx = 0

    async with await NotebookLMClient.from_storage() as client:
        generator = NotebookLMGenerator(
            output_dir=str(output_dir),
            notebook_title=args.notebook,
            client=client,
        )
        await generator._ensure_notebook()

        for doc_idx, item in enumerate(input_items, 1):
            while True:   # regenerate loop
                print(f"\n[{doc_idx}/{len(input_items)}] {item.document_name}")
                print("-" * 40)

                source_id = await generator._add_source(item)

                aspects_str = (
                    "\n".join(f"- {a}" for a in item.aspects_list)
                    if item.aspects_list
                    else "(toàn bộ văn bản)"
                )
                prompt = prompt_template.format(
                    document_name=item.document_name,
                    doc_info_content=item.content,
                    aspects_list=aspects_str,
                )
                # print(prompt)
                print(f"  [Tokens ~{len(prompt) // 4}] Sending to NotebookLM ...")

                result = await generator._ask_with_retry(prompt, source_id)
                # print(result)
                parsed = generator._parse_response(result) if result else None

                # ── sanitize filename: remove chars illegal on Windows/macOS/Linux ──
                safe_name = re.sub(r'[\\/:*?"<>|]', "_", item.document_name)[:50]

                # ── save raw log ──────────────────────────────────────────────
                raw_path = raw_dir / f"doc_{doc_idx:03d}_{safe_name}.json"
                raw_path.write_text(
                    json.dumps(
                        {"document_name": item.document_name, "prompt": prompt,
                         "raw_response": result, "parsed": parsed},
                        ensure_ascii=False, indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"  [Raw]  {raw_path}")

                # ── save flat per-QA JSON ──────────────────────────────────────
                saved = 0
                if parsed and "task1" in parsed and "task2" in parsed:
                    for qa_type in ("task1", "task2"):
                        for entry in parsed.get(qa_type, []):
                            global_idx += 1
                            qa_path = output_dir / f"{global_idx:04d}_{qa_type}.json"
                            qa_path.write_text(
                                json.dumps(entry, ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                            print(f"  [Saved] {qa_path}")
                            saved += 1
                    print(f"  → {saved} QA entries saved (global #{global_idx})")
                else:
                    print("  [Warn] Could not parse task1/task2 from response")

                # if source_id:
                #     await generator._remove_source(source_id)

                # ── next-action prompt ───────────────────────────────────────
                print()
                print("  [r] Re-generate (overwrites this doc's raw log & flat files)")
                print("  [n] Next document")
                print("  [q] Quit")
                while True:
                    choice = input("  > ").strip().lower()
                    if choice in ("r", "n", "q"):
                        break
                    print("  Unknown — use r / n / q")

                if choice == "r":
                    # delete flat files for this round before regenerating
                    for qa_type in ("task1", "task2"):
                        for fp in output_dir.glob(f"????" f"_{qa_type}.json"):
                            pass   # just re-count below; simpler to let user delete
                    break   # regenerate same document
                elif choice == "n":
                    break   # move to next document
                elif choice == "q":
                    print(f"\nDone. {global_idx} total QA entries in {output_dir}")
                    return


if __name__ == "__main__":
    asyncio.run(main())
