"""
Recalculate BERTScore F1 for all models in:
    eval_results/QA_Part2345/eval_results_llm/{model}/row_results_full_pipeline/row_qa_eval_scores.csv

Generated answers are sourced from:
    eval_results/thinh_eval_llm/{model}/row_results_full_pipeline/row_qa_results_offline.csv

Ground truth answers are sourced from:
    qa_dataset/QA_Part2345.csv

After recalculating bert_f1:
  - Writes updated CSV as {original_name}_edited.csv  (original is never touched)
  - Writes per-model qa_metrics_summary.csv (avg of all metrics)
  - Writes a global all_models_summary.csv aggregating all models

Usage:
    uv run python scripts/recalculate_bert_score.py [--dry-run] [--models MODEL [MODEL ...]]
"""

import argparse
import json
import logging
import re
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
_SCORES_DIR = _REPO_ROOT / "eval_results" / "QA_Part2345" / "eval_results_llm"
_OFFLINE_DIR = _REPO_ROOT / "eval_results" / "thinh_eval_llm"
_GT_PATH     = _REPO_ROOT / "qa_dataset" / "QA_Part2345.csv"
_OUT_SUBDIR  = "row_results_full_pipeline"
_SCORES_FILE = "row_qa_eval_scores.csv"

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


# ─── Judge sub-score parsing ──────────────────────────────────────────────────

def _parse_subscores(raw_eval: str) -> dict[str, float | None]:
    """
    Extract judge sub-scores from the raw_eval JSON blob.
    Returns a dict with keys: legal_accuracy, correct_citation,
    completeness, hallucination_citation, structure  (None if missing).
    """
    keys = ["legal_accuracy", "correct_citation", "completeness",
            "hallucination_citation", "structure"]
    result = {k: None for k in keys}

    if not isinstance(raw_eval, str) or not raw_eval.strip():
        return result

    # Strip ```json ... ``` fences if present
    clean = re.sub(r"^```[a-z]*\s*", "", raw_eval.strip(), flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean)

    try:
        data = json.loads(clean)
        scores = data.get("scores", {})
        if isinstance(scores, dict):
            for k in keys:
                v = scores.get(k)
                if v is not None:
                    try:
                        result[k] = float(v)
                    except (TypeError, ValueError):
                        pass
    except Exception:
        pass

    return result


# ─── Core logic ───────────────────────────────────────────────────────────────

def recalculate_bert_f1_for_model(
    model_name: str,
    metric_bertscore,
    bertscore_tokenizer,
    bertscore_limit: int | None,
    gt_map: dict[int, str],
    dry_run: bool = False,
) -> dict | None:
    """
    Recalculate bert_f1 for one model's row_qa_eval_scores.csv.

    Returns a summary dict (avg metrics) or None on failure.
    """
    scores_path  = _SCORES_DIR / model_name / _OUT_SUBDIR / _SCORES_FILE
    offline_path = _OFFLINE_DIR / model_name / _OUT_SUBDIR / "row_qa_results_offline.csv"

    if not scores_path.exists():
        log.warning("[%s] Scores file not found: %s", model_name, scores_path)
        return None

    if not offline_path.exists():
        log.warning("[%s] Offline results not found: %s", model_name, offline_path)
        return None

    log.info("[%s] Loading scores from %s", model_name, scores_path)
    df_scores  = pd.read_csv(scores_path)
    df_offline = pd.read_csv(offline_path)

    # Build id → generated_answer lookup
    gen_map: dict[int, str] = {}
    for row in df_offline.itertuples(index=False):
        gen_map[int(row.id)] = str(getattr(row, "generated_answer", "") or "")

    log.info("[%s] Recalculating bert_f1 for %d rows …", model_name, len(df_scores))

    new_bert_f1_vals: list[float] = []

    for row in df_scores.itertuples(index=False):
        qid = int(row.id)
        pred_text = gen_map.get(qid, "")
        ref_text  = gt_map.get(qid, "")

        bert_f1 = 0.0
        if pred_text and ref_text and not pred_text.startswith("ERROR:"):
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
                log.warning("[%s] BERTScore failed for QID %s: %s", model_name, qid, e)

        new_bert_f1_vals.append(bert_f1)

    # Update the dataframe
    df_scores = df_scores.copy()
    df_scores["bert_f1"] = new_bert_f1_vals

    # Output path: same dir, same stem + "_edited"
    edited_path = scores_path.with_stem(scores_path.stem + "_edited")

    if not dry_run:
        df_scores.to_csv(edited_path, index=False)
        log.info("[%s] Wrote updated scores → %s", model_name, edited_path)
    else:
        log.info("[%s] (dry-run) Would write updated scores → %s", model_name, edited_path)

    # ── Build per-row numeric summary ────────────────────────────────────────

    summary_rows: list[dict] = []
    for row in df_scores.itertuples(index=False):
        raw_eval = getattr(row, "raw_eval", "") or ""
        subscores = _parse_subscores(str(raw_eval))
        r = {
            "score":                  _safe_float(row.score),
            "bleu":                   _safe_float(row.bleu),
            "rougeL":                 _safe_float(row.rougeL),
            "meteor":                 _safe_float(row.meteor),
            "bert_f1":                _safe_float(row.bert_f1),
            "legal_accuracy":         subscores.get("legal_accuracy"),
            "correct_citation":       subscores.get("correct_citation"),
            "completeness":           subscores.get("completeness"),
            "hallucination_citation": subscores.get("hallucination_citation"),
            "structure":              subscores.get("structure"),
        }
        summary_rows.append(r)

    df_sum = pd.DataFrame(summary_rows)

    metric_cols = [
        "score", "bleu", "rougeL", "meteor", "bert_f1",
        "legal_accuracy", "correct_citation", "completeness",
        "hallucination_citation", "structure",
    ]

    agg: dict[str, float] = {"model": model_name}
    for col in metric_cols:
        if col in df_sum.columns:
            vals = pd.to_numeric(df_sum[col], errors="coerce")
            agg[f"avg_{col}"] = round(float(vals.mean(skipna=True)), 4)
        else:
            agg[f"avg_{col}"] = float("nan")

    # Write per-model summary
    per_model_summary_path = _SCORES_DIR / model_name / _OUT_SUBDIR / "qa_metrics_summary.csv"
    if not dry_run:
        pd.DataFrame([agg]).to_csv(per_model_summary_path, index=False)
        log.info("[%s] Per-model summary → %s", model_name, per_model_summary_path)

    # Log averages
    log.info("[%s] Averages:", model_name)
    for k, v in agg.items():
        if k != "model":
            log.info("  %-30s: %.4f", k, v if not pd.isna(v) else 0.0)

    return agg


def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Recalculate BERTScore F1 for QA_Part2345 LLM evals")
    parser.add_argument(
        "--models", nargs="*",
        help="Specific model folder names to process (default: all under eval_results_llm/)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute new scores but do not write changes to disk"
    )
    args = parser.parse_args()

    # ── Discover models ──────────────────────────────────────────────────────
    if args.models:
        model_names = args.models
    else:
        model_names = sorted(
            d.name for d in _SCORES_DIR.iterdir()
            if d.is_dir() and (d / _OUT_SUBDIR / _SCORES_FILE).exists()
        )

    if not model_names:
        log.error("No model directories found under %s", _SCORES_DIR)
        sys.exit(1)

    log.info("Models to process: %s", model_names)

    # ── Load ground truth ────────────────────────────────────────────────────
    if not _GT_PATH.exists():
        log.error("Ground truth file not found: %s", _GT_PATH)
        sys.exit(1)

    df_gt = pd.read_csv(_GT_PATH)
    gt_map: dict[int, str] = {int(r.id): str(r.answer) for r in df_gt.itertuples(index=False)}
    log.info("Loaded %d ground-truth entries from %s", len(gt_map), _GT_PATH)

    # ── Load BERTScore metric and truncation helpers ──────────────────────────
    log.info("Loading BERTScore metric …")
    metric_bertscore = hf_evaluate.load("bertscore")

    log.info("Loading BERTScore tokenizer for truncation (%s) …", _BERTSCORE_MODEL_TYPE)
    bertscore_tokenizer, bertscore_limit = _get_bertscore_truncation(_BERTSCORE_MODEL_TYPE)

    if vi_tokenize:
        log.info("Vietnamese word segmentation (pyvi.ViTokenizer) enabled.")
    else:
        log.warning("pyvi not found – Vietnamese tokenization disabled.")

    # ── Process each model ───────────────────────────────────────────────────
    all_summaries: list[dict] = []

    for model_name in model_names:
        log.info("=" * 60)
        log.info("Processing model: %s", model_name)
        summary = recalculate_bert_f1_for_model(
            model_name=model_name,
            metric_bertscore=metric_bertscore,
            bertscore_tokenizer=bertscore_tokenizer,
            bertscore_limit=bertscore_limit,
            gt_map=gt_map,
            dry_run=args.dry_run,
        )
        if summary is not None:
            all_summaries.append(summary)

    # ── Write global summary CSV ─────────────────────────────────────────────
    if all_summaries:
        global_summary_path = _SCORES_DIR / "all_models_summary.csv"
        df_global = pd.DataFrame(all_summaries)

        # Reorder: model first, then avg_* cols alphabetically
        avg_cols = sorted(c for c in df_global.columns if c.startswith("avg_"))
        df_global = df_global[["model"] + avg_cols]

        if not args.dry_run:
            df_global.to_csv(global_summary_path, index=False)
            log.info("=" * 60)
            log.info("Global summary → %s", global_summary_path)
        else:
            log.info("(dry-run) Global summary would be written to: %s", global_summary_path)

        log.info("\n%s", df_global.to_string(index=False))
    else:
        log.warning("No summaries produced.")


if __name__ == "__main__":
    main()
