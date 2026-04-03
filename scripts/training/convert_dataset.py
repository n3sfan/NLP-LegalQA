"""
Vietnamese Legal QA dataset formatter → Qwen ChatML / instruction-tuning format.

Handles the train1.json structure with 3 tasks:
  task1 – Yes/No binary relevance judgment (legal_document relevance check)
  task2 – Multiple-choice QA with reasoning trace
  task3 – Open-ended QA with structured reasoning (deductive syllogism)

Each sample is written as a single text file containing Qwen's ChatML conversation:

    <|im_start|>system
    {system_prompt}<|im_end|>
    <|im_start|>user
    {formatted_input}<|im_end|>
    <|im_start|>assistant
    {formatted_output}<|im_end|>

The user message wraps the question; the assistant message contains the
answer (and for task2/task3 the <answer>…</answer> block).
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Bạn là một trợ lý pháp lý chuyên nghiệp, được huấn luyện trên dữ liệu pháp luật Việt Nam. "
    "Nhiệm vụ của bạn là trả lời câu hỏi pháp lý một cách chính xác, đầy đủ và có suy luận rõ ràng. "
    "Hãy đưa ra câu trả lời dựa trên các quy định của pháp luật Việt Nam hiện hành. "
    "Luôn sử dụng định dạng <answer>...</answer> để đóng khung câu trả lời cuối cùng."
)

# ── ChatML tokens ─────────────────────────────────────────────────────────────
IM_START = "<|im_start|>"
IM_END   = "<|im_end|>"
EOT      = "<|eot_id|>"          # sentence-level end-of-turn (Qwen3 style)

# ── Helper: strip think tags from raw reasoning text ─────────────────────────
def strip_think(text: str) -> str:
    """Remove Qwen/QwQ <think>…</think> delimiters, keep inner content."""
    return re.sub(r"</?think[^>]*>", "", text).strip()

# ── Task-specific formatters ───────────────────────────────────────────────────

def format_task1(sample: dict[str, Any]) -> tuple[str, str]:
    """
    Binary relevance: does the legal document answer the question?
    Output: Yes/No with reasoning.
    """
    legal_doc  = sample.get("legal_document", "")
    sub_q      = sample.get("specific_question", "")
    question   = sample.get("question", "")
    choices    = sample.get("choices", [])
    answer_idx = int(sample.get("answer", 0))
    reasoning  = strip_think(sample.get("reasoning", ""))

    # Build choice text
    choice_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(choices))
    answer_text = choices[answer_idx] if answer_idx < len(choices) else "Không"

    user_msg = (
        f"Tài liệu pháp lý:\n{legal_doc}\n\n"
        f"Câu hỏi cụ thể: {sub_q}\n\n"
        f"Câu hỏi: {question}\n"
        f"Các lựa chọn:\n{choice_text}\n\n"
        f"Hãy phân tích và đưa ra câu trả lời."
    )

    assistant_msg = (
        f"{reasoning}\n\n"
        f"<answer>{answer_text}</answer>"
    )
    return user_msg, assistant_msg


def format_task2(sample: dict[str, Any]) -> tuple[str, str]:
    """
    Multiple-choice legal QA with reasoning.
    Output: chosen option + full reasoning trace.
    """
    question   = sample.get("question", "")
    choices    = sample.get("choices", [])
    answer_idx = int(sample.get("answer", 0))
    reasoning  = strip_think(sample.get("reasoning", ""))

    choice_text = "\n".join(f"  {chr(65+i)}. {c}" for i, c in enumerate(choices))
    answer_text = f"{chr(65+answer_idx)}. {choices[answer_idx]}" if answer_idx < len(choices) else "Không xác định"

    user_msg = (
        f"Câu hỏi: {question}\n\n"
        f"Các lựa chọn:\n{choice_text}\n\n"
        f"Hãy phân tích từng lựa chọn và đưa ra câu trả lời đúng."
    )

    assistant_msg = (
        f"{reasoning}\n\n"
        f"<answer>{answer_text}</answer>"
    )
    return user_msg, assistant_msg


def format_task3(sample: dict[str, Any]) -> tuple[str, str]:
    """
    Open-ended QA with structured deductive reasoning.
    Output: answer with premises + conclusion in Vietnamese legal style.
    """
    question  = sample.get("question", "")
    answer    = strip_think(sample.get("answer", ""))
    # task3 samples have no separate "reasoning" field; answer itself contains reasoning
    # We keep the answer as-is (it already has structured reasoning + <answer> tag)

    user_msg = f"Câu hỏi: {question}\n\nHãy phân tích và đưa ra câu trả lời dựa trên các quy định pháp luật."

    # The answer field already contains structured reasoning; strip any embedded <answer> tags
    # since we'll add our own outer wrapper
    clean_answer = re.sub(r"</?answer[^>]*>", "", answer).strip()

    assistant_msg = f"{clean_answer}\n\n<answer>{_extract_final_answer(clean_answer)}</answer>"
    return user_msg, assistant_msg


def _extract_final_answer(text: str) -> str:
    """Pull the final verdict / conclusion from structured reasoning text."""
    lines = text.strip().split("\n")
    # Find the last non-empty line that looks like a conclusion
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith("*") and "tiền đề" not in line.lower():
            return line
    return text.strip()[:300]


def sample_to_chatml(sample: dict[str, Any]) -> str:
    """Convert any task sample to a single ChatML-format string."""
    task = None
    if "legal_document" in sample:
        task = "task1"
    elif "choices" in sample and "answer" in sample:
        task = "task2"
    elif "answer" in sample:
        task = "task3"

    if task == "task1":
        user_inp, assistant_out = format_task1(sample)
    elif task == "task2":
        user_inp, assistant_out = format_task2(sample)
    else:
        user_inp, assistant_out = format_task3(sample)

    return (
        f"{IM_START}system\n{SYSTEM_PROMPT}{IM_END}\n"
        f"{IM_START}user\n{user_inp}{IM_END}\n"
        f"{IM_START}assistant\n{assistant_out}{IM_END}\n"
        f"{EOT}"
    )


# ── Main conversion ───────────────────────────────────────────────────────────

def convert_file(input_path: str, output_path: str) -> int:
    """
    Read train1.json, convert all samples to ChatML .jsonl, write to output_path.
    Returns the number of samples written.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for task_key in ["task1", "task2", "task3"]:
            samples = data.get(task_key, [])
            for sample in samples:
                chatml = sample_to_chatml(sample)
                fout.write(json.dumps({"text": chatml}, ensure_ascii=False) + "\n")
                count += 1

    return count


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_dataset.py <input.json> <output.jsonl>")
        sys.exit(1)

    count = convert_file(sys.argv[1], sys.argv[2])
    print(f"✓ Converted {count} samples → {sys.argv[2]}")
