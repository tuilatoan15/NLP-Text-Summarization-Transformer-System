"""
abstractive.py — Optimized abstractive summarization wrappers.

Optimization summary vs old version
─────────────────────────────────────
1. Models are NO LONGER loaded per-request.
   AbstractiveSummarizer.load() now delegates to model_loader.get_loaded_model()
   which returns a pre-warmed, singleton model — zero disk I/O after startup.

2. GPU-first inference:
   • torch.inference_mode() (faster than no_grad for inference)
   • torch.cuda.amp.autocast() with fp16 when GPU supports it
   • log_vram_usage() before/after generation for debugging

3. Optimized generate() parameters:
   • max_new_tokens (not max_length) avoids input-length miscounting
   • num_beams=4 (quality/speed balance)
   • repetition_penalty=1.15 to suppress loops
   • no_repeat_ngram_size=3
   • early_stopping=True

4. ViT5 special-token cleanup:
   • strip <extra_id_*> SentencePiece artifacts
   • UTF-8 safe decode (skip_special_tokens=True, clean_up_tokenization_spaces=False)
   • is_probably_bad_generation() guard with detailed logging

5. Chunked summarization for long documents:
   • splits by sentence, respects MAX_INPUT_TOKENS budget
   • merges partials with a recursive final pass
"""

from __future__ import annotations

import os
import time
from typing import Optional

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import torch

from src import config
from src.model_loader import get_device, get_loaded_model, is_fp16
from src.model_registry import ABSTRACTIVE_ALGORITHMS, AlgorithmConfig
from src.preprocess import (
    clean_generated_summary,
    clean_text,
    is_probably_bad_generation,
    split_sentences,
)
from src.utils import log_vram_usage, logger


# ─────────────────────────── Internal helpers ───────────────────────────────

def _algorithm_from_key(key: str) -> AlgorithmConfig:
    """Resolve an algorithm key → AlgorithmConfig, with alias handling."""
    key = (key or "").strip().lower()
    aliases = {"phobart": "bartpho", "pho-bart": "bartpho", "bart": "bartpho"}
    key = aliases.get(key, key)
    if key in ABSTRACTIVE_ALGORITHMS:
        return ABSTRACTIVE_ALGORITHMS[key]
    # Fallback: wrap an unknown HF model name
    return AlgorithmConfig(
        key=key.replace("/", "__"),
        name=key,
        group="abstractive",
        model_name=key,
        description="Custom HuggingFace seq2seq model.",
    )


def _prefix_text(key: str, text: str) -> str:
    """Add task prefix required by T5-family models."""
    if key in {"vit5", "mt5"}:
        return f"summarize: {text}"
    return text


def _chunk_text(text: str, max_words_per_chunk: int) -> list[str]:
    """Split text into sentence-aligned chunks that fit the model's token budget."""
    sentences = split_sentences(text)
    if not sentences:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        w = len(sentence.split())
        if current and current_words + w > max_words_per_chunk:
            chunks.append(" ".join(current))
            current, current_words = [], 0
        current.append(sentence)
        current_words += w

    if current:
        chunks.append(" ".join(current))

    return chunks[:8]  # hard cap: never process more than 8 chunks


# ─────────────────────────── Core inference ────────────────────────────────

def _generate_one(
    key: str,
    text: str,
    max_new_tokens: int,
    min_new_tokens: int,
    num_beams: int,
) -> str:
    """
    Run one forward pass through a loaded Transformer model.

    The model and tokenizer are fetched from the singleton registry —
    no disk I/O, no model instantiation, no repeated .to(device) calls.
    """
    loaded = get_loaded_model(key)
    model = loaded.model
    tokenizer = loaded.tokenizer
    device = loaded.device
    use_fp16 = loaded.fp16

    prefixed = _prefix_text(key, text)
    encoded = tokenizer(
        prefixed,
        return_tensors="pt",
        truncation=True,
        max_length=config.MAX_INPUT_TOKENS,
        padding=False,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    log_vram_usage(f"pre_generate_{key}")
    t_start = time.perf_counter()

    # torch.inference_mode is faster than no_grad — disables all autograd bookkeeping
    with torch.inference_mode():
        if use_fp16 and device.type == "cuda":
            # autocast lets the GPU run matmuls in fp16 while keeping
            # numerically sensitive ops in fp32 automatically
            with torch.cuda.amp.autocast(dtype=torch.float16):
                generated_ids = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    num_beams=num_beams,
                    no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
                    repetition_penalty=1.15,
                    length_penalty=1.0,
                    early_stopping=True,
                    do_sample=False,
                )
        else:
            generated_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                num_beams=num_beams,
                no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
                repetition_penalty=1.15,
                length_penalty=1.0,
                early_stopping=True,
                do_sample=False,
            )

    elapsed = time.perf_counter() - t_start
    logger.debug("[%s] generate() took %.3f s", key, elapsed)
    log_vram_usage(f"post_generate_{key}")

    # Decode — skip_special_tokens removes <pad>, </s>, <unk> etc.
    # clean_up_tokenization_spaces=False preserves Vietnamese spacing exactly
    decoded = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    raw = decoded[0] if decoded else ""
    summary = clean_generated_summary(raw)

    if is_probably_bad_generation(summary):
        logger.warning(
            "[%s] Bad generation detected (len=%d, preview=%r) — returning empty",
            key, len(summary), summary[:80],
        )
        return ""

    return summary


# ─────────────────────────── Public summarize function ─────────────────────

def abstractive_summarize_key(
    key: str,
    text: str,
    max_output_length: int = config.MAX_OUTPUT_LENGTH,
    min_output_length: int = config.MIN_OUTPUT_LENGTH,
    num_beams: int = config.NUM_BEAMS,
) -> str:
    """
    Summarize *text* using the model identified by *key* (e.g. 'vit5').

    For long texts, the input is split into chunks, each chunk summarised,
    then the partial summaries are merged in a final recursive pass.
    """
    text = clean_text(text, aggressive=True)
    if not text:
        return ""

    # Estimate whether the text needs chunking (word-based heuristic)
    max_words = max(180, int(config.MAX_INPUT_TOKENS * 0.65))
    if len(text.split()) > max_words:
        chunks = _chunk_text(text, max_words)
        logger.debug("[%s] Long text — splitting into %d chunks", key, len(chunks))

        partials = [
            _generate_one(
                key,
                chunk,
                max_new_tokens=max(48, max_output_length // 2),
                min_new_tokens=min(24, min_output_length),
                num_beams=num_beams,
            )
            for chunk in chunks
        ]
        merged = " ".join(p for p in partials if p)

        # If merged partials are still too long, do one final compression pass
        if len(merged.split()) > max_output_length:
            return abstractive_summarize_key(
                key, merged,
                max_output_length=max_output_length,
                min_output_length=min_output_length,
                num_beams=num_beams,
            )
        return clean_generated_summary(merged)

    return _generate_one(key, text, max_output_length, min_output_length, num_beams)


# ─────────────────────────── Compatibility shim ────────────────────────────
# The old AbstractiveSummarizer class is preserved as a thin wrapper so that
# anything still calling get_summarizer() / AbstractiveSummarizer.summarize()
# continues to work without code changes.

class AbstractiveSummarizer:
    """
    Thin compatibility wrapper around abstractive_summarize_key().

    All heavy lifting (model loading, GPU scheduling, fp16, etc.) happens in
    model_loader.py / _generate_one().  This class is kept only so that
    existing call-sites in dashboard.py / tests do not need to change.
    """

    def __init__(
        self,
        model_name: str = "vit5",
        local_model_dir: Optional[str] = None,
        max_input_tokens: int = config.MAX_INPUT_TOKENS,
        max_output_length: int = config.MAX_OUTPUT_LENGTH,
        min_output_length: int = config.MIN_OUTPUT_LENGTH,
        num_beams: int = config.NUM_BEAMS,
        no_repeat_ngram_size: int = config.NO_REPEAT_NGRAM_SIZE,
    ) -> None:
        self.algorithm = _algorithm_from_key(model_name)
        self.key = self.algorithm.key
        self.max_input_tokens = max_input_tokens
        self.max_output_length = max_output_length
        self.min_output_length = min_output_length
        self.num_beams = num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size

    # Keep the old .load() signature — it's now effectively a no-op because
    # model_loader already holds the model.
    def load(self) -> None:
        _ = get_loaded_model(self.key)

    def is_loaded(self) -> bool:
        from src.model_loader import _registry
        return _registry.is_loaded(self.key)

    @property
    def device(self) -> torch.device:
        return get_device()

    def summarize(
        self,
        text: str,
        max_output_length: Optional[int] = None,
        min_output_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        chunk_long_text: bool = True,
    ) -> str:
        return abstractive_summarize_key(
            self.key,
            text,
            max_output_length=max_output_length or self.max_output_length,
            min_output_length=min_output_length or self.min_output_length,
            num_beams=num_beams or self.num_beams,
        )

    def explain_tokens(self, source_text: str, summary: str, limit: int = 40) -> list[dict]:
        source_tokens = _tokenize_for_importance(source_text)
        summary_tokens = _tokenize_for_importance(summary)
        if not summary_tokens:
            return []
        counts = {token: source_tokens.count(token) for token in set(source_tokens)}
        max_count = max(counts.values()) if counts else 1
        return [
            {
                "token": token,
                "importance": round((counts.get(token, 0) / max_count) if max_count else 0.0, 4),
            }
            for token in summary_tokens[:limit]
        ]


def _tokenize_for_importance(text: str) -> list[str]:
    from src.preprocess import tokenize_words
    return tokenize_words(text, remove_stopwords=True)


# ── Backward-compatible module-level helpers ────────────────────────────────

_global_summarizers: dict[str, AbstractiveSummarizer] = {}


def get_summarizer(
    model_name: str = "vit5",
    local_model_dir: Optional[str] = None,
) -> AbstractiveSummarizer:
    key = _algorithm_from_key(model_name).key
    cache_key = f"{key}|{local_model_dir or ''}"
    if cache_key not in _global_summarizers:
        _global_summarizers[cache_key] = AbstractiveSummarizer(
            model_name=key,
            local_model_dir=local_model_dir,
        )
    return _global_summarizers[cache_key]


def abstractive_summarize(
    text: str,
    max_output_length: int = config.MAX_OUTPUT_LENGTH,
    min_output_length: int = config.MIN_OUTPUT_LENGTH,
    num_beams: int = config.NUM_BEAMS,
    local_model_dir: Optional[str] = None,
    model_name: str = "vit5",
) -> str:
    key = _algorithm_from_key(model_name).key
    return abstractive_summarize_key(
        key, text,
        max_output_length=max_output_length,
        min_output_length=min_output_length,
        num_beams=num_beams,
    )
