"""
Merge evaluation results from QA_Part2, QA_Part3, QA_Part4, and QA_Part5 LLM evaluations,
and recalculate BERTScore F1 for the merged files using segmented Vietnamese text.

Usage:
    uv run python scripts/merge_parts_bert_score.py [--files FILE1 [FILE2 ...]]

Default files merged:
    - geminiflash_fewshot.csv
    - gemma4_fewshot.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import evaluate as hf_evaluate

try:
    from transformers import AutoConfig, AutoTokenizer
except ImportError:
    AutoConfig = None
    AutoTokenizer = None

try:
    from pyvi.ViTokenizer import tokenize as vi_tokenize
except ImportError:
    vi_tokenize = None
    logging.warning("pyvi not installed – skipping Vietnamese word segmentation before BERTScore.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EVAL_DIR  = _REPO_ROOT / "eval_results"
_OUT_DIR   = _EVAL_DIR / "QA_Part2345" / "eval_results_llm"

_PARTS = [2, 3, 4, 5]

# ─── BERTScore config (mirrors src/llm/eval_voter.py) ────────────────────────

_BERTSCORE_MODEL_TYPE = "bkai-foundation-models/vietnamese-bi-encoder"
_BERTSCORE_NUM_LAYERS = 9

# Tokenizer truncation cache
_trunc_cache: dict[str, tuple] = {}


def _get_bertscore_truncation(model_type: str) -> tuple:
    """Load tokenizer + safe token limit for the encoder (cached)."""
    if model_type in _trunc_cache:
        return _trunc_cache[model_type]

    if AutoTokenizer is None or AutoConfig is None:
        _trunc_cache[model_type] = (None, None)
        return _trunc_cache[model_type]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        config    = AutoConfig.from_pretrained(model_type)
        max_pos   = getattr(config, "max_position_embeddings", None)
        limit     = max_pos - 15 if isinstance(max_pos, int) and max_pos > 15 else None
        _trunc_cache[model_type] = (tokenizer, limit)
    except Exception as e:
        log.warning("Could not load BERTScore truncation for %s: %s", model_type, e)
        _trunc_cache[model_type] = (None, None)

    return _trunc_cache[model_type]


def _truncate(text: str, tokenizer, limit: int | None) -> str:
    """Trim text to the encoder's safe token window."""
    if not text or tokenizer is None or not limit or limit <= 0:
        return text
    try:
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=limit)
        return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except Exception:
        return text


def _vi_tok(text: str) -> str:
    """Apply Vietnamese word segmentation if pyvi is available."""
    if vi_tokenize and text:
        return vi_tokenize(text)
    return text


def merge_and_calculate_file(
    filename: str,
    metric_bertscore,
    bertscore_tokenizer,
    bertscore_limit: int | None,
) -> None:
    """Merge part CSVs for filename, compute bert_f1, and save the merged file."""
    log.info("=" * 60)
    log.info("Merging files named '%s' across Parts %s ...", filename, _PARTS)

    dfs = []
    for part in _PARTS:
        part_path = _EVAL_DIR / f"QA_Part{part}" / "eval_results_llm" / filename
        if not part_path.exists():
            log.warning("File not found for Part %d: %s", part, part_path)
            continue
        log.info("Loading %s", part_path)
        df_part = pd.read_csv(part_path)
        
        # Apply offset to the ID column if present to ensure unique sequential IDs 1-200
        if "id" in df_part.columns:
            df_part = df_part.copy()
            df_part["id"] = pd.to_numeric(df_part["id"], errors="coerce")
            if part == 4:
                df_part["id"] += 100
                log.info("Offset Part 4 IDs by +100")
            elif part == 5:
                df_part["id"] += 150
                log.info("Offset Part 5 IDs by +150")
                
        dfs.append(df_part)

    if not dfs:
        log.error("No source files found for %s! Skipping.", filename)
        return

    # Concatenate and sort by id
    df_merged = pd.concat(dfs, ignore_index=True)
    if "id" in df_merged.columns:
        df_merged["id"] = pd.to_numeric(df_merged["id"], errors="coerce")
        df_merged = df_merged.sort_values(by="id").reset_index(drop=True)

    log.info("Merged dataframe has %d rows.", len(df_merged))

    if "ground_truth" not in df_merged.columns or "generated_answer" not in df_merged.columns:
        log.error("Merged file is missing 'ground_truth' or 'generated_answer' columns! Skipping BERTScore.")
        return

    # Compute BERTScore F1 for each row
    log.info("Recalculating bert_f1 using %s model ...", _BERTSCORE_MODEL_TYPE)
    bert_f1_vals = []

    for idx, row in enumerate(df_merged.itertuples()):
        pred_text = getattr(row, "generated_answer", "") or ""
        ref_text  = getattr(row, "ground_truth", "") or ""

        # Make sure texts are valid strings and not error indicators
        pred_text = str(pred_text)
        ref_text  = str(ref_text)

        bert_f1 = 0.0
        if pred_text.strip() and ref_text.strip() and not pred_text.startswith("ERROR:"):
            try:
                bert_pred = _truncate(_vi_tok(pred_text), bertscore_tokenizer, bertscore_limit)
                bert_ref  = _truncate(_vi_tok(ref_text),  bertscore_tokenizer, bertscore_limit)
                result = metric_bertscore.compute(
                    predictions=[bert_pred],
                    references=[bert_ref],
                    model_type=_BERTSCORE_MODEL_TYPE,
                    num_layers=_BERTSCORE_NUM_LAYERS,
                )
                bert_f1 = round(result["f1"][0], 4) if result else 0.0
            except Exception as e:
                log.warning("Row %d (QID %s) BERTScore calculation failed: %s", idx, getattr(row, "id", "N/A"), e)

        bert_f1_vals.append(bert_f1)

    # Insert bert_f1 right after meteor if present, otherwise append
    if "bert_f1" in df_merged.columns:
        df_merged["bert_f1"] = bert_f1_vals
    else:
        if "meteor" in df_merged.columns:
            ins_idx = df_merged.columns.get_loc("meteor") + 1
            df_merged.insert(ins_idx, "bert_f1", bert_f1_vals)
        else:
            df_merged["bert_f1"] = bert_f1_vals

    # Save output to destination
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / filename
    df_merged.to_csv(out_path, index=False)
    log.info("Saved merged file with recalculated bert_f1 → %s", out_path)

    # Calculate and display metrics summary
    numeric_cols = [
        "bleu", "rougeL", "meteor", "bert_f1",
        "judge_legal_accuracy", "judge_correct_citation", "judge_completeness",
        "judge_hallucination_citation", "judge_structure", "judge_overall_score",
        "gen_time", "judge_time"
    ]

    log.info("Averages for merged %s:", filename)
    summary_data = {}
    for col in numeric_cols:
        if col in df_merged.columns:
            vals = pd.to_numeric(df_merged[col], errors="coerce")
            mean_val = float(vals.mean(skipna=True))
            summary_data[col] = round(mean_val, 4)
            log.info("  %-30s: %.4f", f"avg_{col}", mean_val if not pd.isna(mean_val) else 0.0)

    log.info("\nMerged Summary Row:\n%s\n", pd.DataFrame([summary_data]).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge part QA evals and compute BERT F1 scores")
    parser.add_argument(
        "--files", nargs="+",
        default=["geminiflash_fewshot.csv", "gemma4_fewshot.csv"],
        help="Filenames to merge across QA_Parts (default: geminiflash_fewshot.csv, gemma4_fewshot.csv)"
    )
    args = parser.parse_args()

    # Load BERTScore metric and truncation helpers
    log.info("Loading BERTScore metric ...")
    metric_bertscore = hf_evaluate.load("bertscore")

    log.info("Loading BERTScore tokenizer for truncation (%s) ...", _BERTSCORE_MODEL_TYPE)
    bertscore_tokenizer, bertscore_limit = _get_bertscore_truncation(_BERTSCORE_MODEL_TYPE)

    if vi_tokenize:
        log.info("Vietnamese word segmentation (pyvi.ViTokenizer) enabled.")
    else:
        log.warning("pyvi not found – Vietnamese tokenization disabled.")

    for filename in args.files:
        merge_and_calculate_file(
            filename=filename,
            metric_bertscore=metric_bertscore,
            bertscore_tokenizer=bertscore_tokenizer,
            bertscore_limit=bertscore_limit,
        )


if __name__ == "__main__":
    main()
