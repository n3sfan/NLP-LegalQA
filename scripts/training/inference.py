"""
Inference script — load QLoRA adapter + base model and chat with the model.

Usage:
    python scripts/training/inference.py \
        --base_model Qwen/Qwen2.5-4B \
        --adapter_path outputs/qlora_legal_qa/final/ \
        --max_new_tokens 512
"""

import argparse
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM_PROMPT = (
    "Bạn là một trợ lý pháp lý chuyên nghiệp, được huấn luyện trên dữ liệu pháp luật Việt Nam. "
    "Nhiệm vụ của bạn là trả lời câu hỏi pháp lý một cách chính xác, đầy đủ và có suy luận rõ ràng. "
    "Luôn sử dụng định dạng <answer>...</answer> để đóng khung câu trả lời cuối cùng."
)


def build_prompt(question: str) -> str:
    """Wrap user question in ChatML format."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nCâu hỏi: {question}\n\n"
        f"Hãy phân tích và đưa ra câu trả lời dựa trên các quy định pháp luật Việt Nam.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Chat with QLoRA-finetuned Qwen model")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-4B")
    parser.add_argument("--adapter_path", required=True,
                        help="Path to saved PeftModel (final/)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive chat loop")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32

    print("[INFO] Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Loading base model (4-bit) …")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    print("[INFO] Loading LoRA adapter …")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    print("[INFO] Ready. Type your legal question and press Enter.")
    print("        Use Ctrl+C to exit.\n")

    if args.interactive:
        history = []

        def format_history():
            msgs = []
            for role, content in history:
                msgs.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            return "\n".join(msgs)

        while True:
            try:
                q = input("Bạn > ").strip()
                if not q:
                    continue
                history.append(("user", f"Câu hỏi: {q}\n\nHãy phân tích và đưa ra câu trả lời dựa trên các quy định pháp luật Việt Nam."))
                prompt = (
                    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                    f"{format_history()}\n"
                    f"<|im_start|>assistant\n"
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                )
                response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
                response = response.replace("<|im_end|>", "").strip()
                print(f"\nTrợ lý > {response}\n")
                history.append(("assistant", response))
            except KeyboardInterrupt:
                print("\n[Tail]")
                break
    else:
        # Single-question mode
        q = input("Câu hỏi: ").strip()
        prompt = build_prompt(q)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        response = response.replace("<|im_end|>", "").replace("<|eot_id|>", "").strip()
        print(f"\nĐáp án:\n{response}")


if __name__ == "__main__":
    main()
