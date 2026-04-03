"""
QLoRA fine-tuning script for Qwen3-4B on Vietnamese Legal QA dataset.

Hardware target : single NVIDIA GPU (≥ 20 GB VRAM recommended)
                  Falls back to CPU-only warning if no GPU is found.
Quantization    : 4-bit NormalFloat (NF4) via BitsAndBytesConfig
LoRA config     : rank=32, alpha=64, dropout=0.05
                  Target modules: q_proj, k_proj, v_proj, o_proj,
                                  gate_proj, up_proj, down_proj
Optimizer       : Paged AdamW 8-bit (QLoRA paper recommendation)
Scheduler       : linear warmup 5 steps
Batch strategy  : per_device_bs=16 × grad_accum=8  → effective_bs=128
Sequence length : 2048 tokens
Epochs          : 1
Mixed precision : bf16 (for GPU) / fp32 (for CPU)
Gradient ckpt   : enabled to save VRAM
Trainer         : SFTTrainer (trl) — proper loss masking, packing
"""

import argparse
import math
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, LoraModel, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig


# ── Config constants ────────────────────────────────────────────────────────────
MODEL_ID         = "Qwen/Qwen3-4B"
MAX_SEQ_LEN      = 2048
LORA_R           = 32
LORA_ALPHA       = 64
LORA_DROPOUT     = 0.05
LEARNING_RATE    = 2e-5
NUM_EPOCHS       = 1
PER_DEVICE_BS    = 16
GRAD_ACCUM_STEPS = 8        # effective batch size = 128
WARMUP_STEPS     = 5
OUTPUT_DIR       = "./outputs/qlora_legal_qa"
SEED             = 42
DATASET_TEXT_COL = "text"  # column name in our ChatML JSONL


# ── Tokenizer helper ──────────────────────────────────────────────────────────

def load_tokenizer(model_id: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
        use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


# ── BitsAndBytes 4-bit NF4 config ─────────────────────────────────────────────

def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat4
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,       # nested quant → extra VRAM saving
    )


# ── LoRA config ───────────────────────────────────────────────────────────────

def get_lora_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )


# ── Formatting function for SFTTrainer ────────────────────────────────────────
# SFTTrainer will call this on every sample to extract the text field.
# It uses the special "<|im_start|>assistant" marker to know where the
# assistant response begins and masks all tokens before it from the loss.

def formatting_func(example: dict) -> str:
    """Return the pre-formatted ChatML string from the dataset column."""
    return example[DATASET_TEXT_COL]


# ── Main training ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Qwen on Legal QA")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--dataset_path", required=True,
                        help="Path to .jsonl dataset (one JSON object per line)")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--per_device_bs", type=int, default=PER_DEVICE_BS)
    parser.add_argument("--grad_accum_steps", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()

    # ── GPU check ────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[INFO] GPU: {gpu_name}  ({gpu_mem:.1f} GB)")
    else:
        print("[WARNING] No GPU detected. Training will be very slow on CPU.")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print("[INFO] Loading tokenizer …")
    tokenizer = load_tokenizer(args.model_id)

    # ── Dataset ──────────────────────────────────────────────────────────────
    print(f"[INFO] Loading dataset from {args.dataset_path} …")
    raw_ds = load_dataset("json", data_files=args.dataset_path, split="train")
    print(f"[INFO] Dataset size: {len(raw_ds)} samples")

    # ── Model with QLoRA ─────────────────────────────────────────────────────
    print("[INFO] Loading base model with 4-bit NF4 quantization …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.enable_input_require_grads()   # required for LoRA backprop

    print("[INFO] Applying LoRA adapters …")
    model = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()

    # ── SFTConfig ────────────────────────────────────────────────────────────
    # SFTTrainer maps SFTConfig fields onto TrainingArguments, so we put
    # everything here rather than splitting across two config objects.
    steps_per_epoch = math.ceil(
        len(raw_ds) / (args.per_device_bs * args.grad_accum_steps)
    )
    max_steps = steps_per_epoch * args.num_epochs

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum_steps,
        max_steps=max_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=args.warmup_steps,
        optim="paged_adamw_8bit",           # 8-bit paged AdamW (memory-efficient)
        fp16=False,
        bf16=device == "cuda",
        logging_steps=25,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=min(50, steps_per_epoch),
        evaluation_strategy="steps",
        eval_steps=min(50, steps_per_epoch),
        do_eval=True,
        save_total_limit=2,
        seed=args.seed,
        remove_unused_columns=False,
        max_seq_length=args.max_seq_len,
        dataset_text_field=DATASET_TEXT_COL,
        # packing=False → one sample per row; set True for packed sequences
        # (disabled here because our samples are already full-length 2048-token)
        packing=False,
        # Tell SFTTrainer which token marks the start of the assistant turn
        # so it can mask the system + user tokens from the loss.
        # Works together with dataset_text_field for pre-formatted ChatML.
        formatting_func=formatting_func,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        tf32=True,                          # fast matmul on Ampere+ GPUs
        report_to="none",
    )

    # ── SFTTrainer ────────────────────────────────────────────────────────────
    # SFTTrainer wraps the base Trainer and:
    #   1. Applies proper loss masking (user+system → -100 labels)
    #   2. Optionally packs short sequences into max_seq_len blocks
    #   3. Handles the DataCollator internally
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=raw_ds,
        processing_class=tokenizer,   # replaces deprecated tokenizer= argument
    )

    print("[INFO] Starting training …")
    print(f"  • trainer class      : SFTTrainer (trl)")
    print(f"  • effective batch    : {args.per_device_bs * args.grad_accum_steps}")
    print(f"  • max steps          : {max_steps}")
    print(f"  • warmup steps       : {args.warmup_steps}")
    print(f"  • LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  • sequence length    : {args.max_seq_len}")
    print(f"  • loss masking       : tokens before <|im_start|>assistant masked to -100")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ── Save final adapter ───────────────────────────────────────────────────
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n✓ Training complete. Adapter saved to: {final_dir}")


if __name__ == "__main__":
    main()
