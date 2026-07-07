"""Tokenization statistics for VietNews analytics (ViT5 subword)."""

from __future__ import annotations

from functools import lru_cache
from statistics import mean
from typing import Any

from backend.services.dataset_analysis.cleaner import tokenize_words
from src import config

TOKEN_BATCH_SIZE = 500


@lru_cache(maxsize=1)
def _get_hf_tokenizer():
    try:
        from transformers import AutoTokenizer

        model_name = config.DEFAULT_MODEL_NAME
        # ViT5/T5 fast tokenizer có thể lỗi Unigram trên một số bản transformers
        use_fast = not any(
            x in model_name.lower() for x in ("vit5", "mt5", "t5", "bartpho")
        )
        return AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    except Exception:
        return None


def count_word_tokens(text: str) -> int:
    return len(tokenize_words(text))


def count_subword_tokens(text: str) -> int | None:
    tokenizer = _get_hf_tokenizer()
    if tokenizer is None:
        return None
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return None


def _percentile(sorted_vals: list[int], p: float) -> int:
    if not sorted_vals:
        return 0
    idx = int(p * (len(sorted_vals) - 1))
    return sorted_vals[idx]


def token_stats_for_texts(
    texts: list[str],
    *,
    sample_size: int | None = None,
    max_input_tokens: int | None = None,
    max_target_tokens: int | None = None,
) -> dict[str, Any]:
    """Compute word + ViT5 subword stats. ``sample_size=None/0`` → full list."""
    if not texts:
        return {}

    if sample_size is None or sample_size <= 0 or sample_size >= len(texts):
        sample = texts
    else:
        sample = texts[:sample_size]

    word_counts = [count_word_tokens(t) for t in sample]
    subword_counts: list[int] = []
    truncated_input = 0
    truncated_target = 0
    max_in = max_input_tokens or config.MAX_INPUT_TOKENS
    max_tgt = max_target_tokens or config.MAX_TARGET_TOKENS

    for i in range(0, len(sample), TOKEN_BATCH_SIZE):
        batch = sample[i : i + TOKEN_BATCH_SIZE]
        for text in batch:
            sw = count_subword_tokens(text)
            if sw is not None:
                subword_counts.append(sw)
                if sw > max_in:
                    truncated_input += 1
                if sw > max_tgt:
                    truncated_target += 1

    word_sorted = sorted(word_counts)
    sub_sorted = sorted(subword_counts) if subword_counts else []

    result: dict[str, Any] = {
        "word_token_avg": round(mean(word_counts), 2),
        "word_token_min": min(word_counts),
        "word_token_max": max(word_counts),
        "word_token_p95": _percentile(word_sorted, 0.95),
        "word_token_p99": _percentile(word_sorted, 0.99),
        "tokenizer_model": config.DEFAULT_MODEL_NAME,
        "sample_size": len(sample),
        "full_dataset": len(sample) == len(texts),
    }
    if sub_sorted:
        result.update(
            {
                "subword_token_avg": round(mean(sub_sorted), 2),
                "subword_token_min": min(sub_sorted),
                "subword_token_max": max(sub_sorted),
                "subword_token_p95": _percentile(sub_sorted, 0.95),
                "subword_token_p99": _percentile(sub_sorted, 0.99),
                "truncation_estimate_input": {
                    "max_tokens": max_in,
                    "would_truncate": truncated_input,
                    "pct": round(100 * truncated_input / len(sub_sorted), 2),
                },
                "truncation_estimate_target": {
                    "max_tokens": max_tgt,
                    "would_truncate": truncated_target,
                    "pct": round(100 * truncated_target / len(sub_sorted), 2),
                },
            }
        )
    else:
        result["subword_token_avg"] = None
    return result
