#!/usr/bin/env bash
# ============================================================
# QLoRA Fine-tuning Pipeline — Vietnamese Legal QA
#
# Usage:
#   bash scripts/training/run_training.sh
#
# Prerequisites:
#   1. conda activate ml-env     (or your env with pip install -r requirements_training.txt)
#   2. GPU with ≥ 20 GB VRAM recommended (RTX 3090/4090/A100/L40)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data2"

# ── Step 0: Environment check ───────────────────────────────
echo "========================================"
echo "  QLoRA Training Pipeline — Legal QA"
echo "========================================"

echo "[Step 0] Environment check"
if ! command -v python &>/dev/null; then
    echo "[ERROR] python not found. Activate your conda env: conda activate ml-env"
    exit 1
fi

# Check GPU
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEM=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}')")
    echo "  ✓ GPU: $GPU_NAME  ($GPU_MEM GB)"
else
    echo "  ⚠ No GPU detected — training will be extremely slow on CPU."
fi

# ── Step 1: Convert dataset to ChatML JSONL ────────────────
echo ""
echo "[Step 1] Converting dataset → ChatML format"
CONV_OUT="$DATA_DIR/train1.jsonl"

python "$SCRIPT_DIR/convert_dataset.py" \
    "$DATA_DIR/train1.json" \
    "$CONV_OUT"

LINES=$(wc -l < "$CONV_OUT")
echo "  ✓ Converted $LINES samples → $CONV_OUT"

# ── Step 2: Run QLoRA training ─────────────────────────────
echo ""
echo "[Step 2] Starting QLoRA fine-tuning"
echo "  • model    : Qwen/Qwen2.5-4B (or Qwen3-4B)"
echo "  • LoRA r   : 32  | alpha: 64"
echo "  • quant    : 4-bit NF4 + double quant"
echo "  • eff. BS  : 128  (16 × 8)"
echo "  • max seq  : 2048 tokens"
echo "  • epochs   : 1"
echo "  • LR       : 2e-5  | warmup: 5 steps"
echo ""

python "$SCRIPT_DIR/train_qlora.py" \
    --model_id "Qwen/Qwen2.5-4B" \
    --dataset_path "$CONV_OUT" \
    --output_dir "$PROJECT_DIR/outputs/qlora_legal_qa" \
    --max_seq_len 2048 \
    --per_device_bs 16 \
    --grad_accum_steps 8 \
    --num_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_steps 5 \
    --seed 42

echo ""
echo "========================================"
echo "  ✓ Training complete!"
echo "  Adapter saved: $PROJECT_DIR/outputs/qlora_legal_qa/final/"
echo "========================================"
