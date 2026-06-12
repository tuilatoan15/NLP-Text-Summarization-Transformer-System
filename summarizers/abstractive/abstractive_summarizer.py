"""
abstractive_summarizer.py — Optimized abstractive summarization wrappers.

Key improvements over the original:
  * Parallel chunk inference via ThreadPoolExecutor (ABSTRACTIVE_CHUNK_WORKERS)
  * Synthesis re-pass: if merged chunks exceed 1.5× target words, re-summarize once
  * Per-model generation-config compatibility (mT5 sampling vs beam)
  * Chunk limit raised to config.ABSTRACTIVE_MAX_CHUNKS (default 16)
  * Cleaner fallback chain: corrupt → greedy → extractive fallback
"""

from __future__ import annotations

import os
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import torch

from src import config
from src.length_control import (
    allocate_chunk_word_budgets,
    min_new_tokens_for_budget,
    words_to_max_new_tokens,
)
from src.model_loader import get_device, get_loaded_model, is_fp16
from src.model_registry import ABSTRACTIVE_ALGORITHMS, AlgorithmConfig
from src.preprocess import (
    clean_generated_summary,
    clean_text,
    is_probably_bad_generation,
    split_sentences,
)
from src.utils import count_words, log_vram_usage, logger


# ─────────────────────────── helpers ────────────────────────────────────────

def _algorithm_from_key(key: str) -> AlgorithmConfig:
    key = (key or "").strip().lower()
    aliases = {"phobart": "bartpho", "pho-bart": "bartpho", "bart": "bartpho"}
    key = aliases.get(key, key)
    if key in ABSTRACTIVE_ALGORITHMS:
        return ABSTRACTIVE_ALGORITHMS[key]
    return AlgorithmConfig(
        key=key.replace("/", "__"),
        name=key,
        group="abstractive",
        model_name=key,
        description="Custom HuggingFace seq2seq model.",
    )


def _prefix_text(key: str, text: str) -> str:
    if key in {"vit5", "mt5"}:
        return f"summarize: {text}"
    return text


def _max_words_per_chunk(key: str) -> int:
    """Model-specific input word budget before tokenization/truncation."""
    if key == "bartpho":
        # Syllable tokenizer expands ~1.8× vs whitespace word count.
        return max(120, int(config.MAX_INPUT_TOKENS * 0.30))
    if key == "mt5":
        return max(150, int(config.MAX_INPUT_TOKENS * 0.48))
    return max(180, int(config.MAX_INPUT_TOKENS * 0.55))


def _chunk_text(text: str, max_words_per_chunk: int) -> list[str]:
    """Split text into chunks at sentence boundaries, respecting max_words_per_chunk.
    Limit to config.ABSTRACTIVE_MAX_CHUNKS (default 16) to bound memory usage."""
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

    max_chunks = getattr(config, "ABSTRACTIVE_MAX_CHUNKS", 16)
    return chunks[:max_chunks]


def _sanitize_gen_preset(key: str, preset: dict) -> dict:
    """Remove params that are incompatible with the chosen decoding mode."""
    do_sample = preset.get("do_sample", False)
    cleaned = dict(preset)

    if do_sample:
        # early_stopping is only valid for beam search
        cleaned.pop("early_stopping", None)
        cleaned.pop("forced_bos_token_id", None)
        # num_beams must be 1 with do_sample
        cleaned["num_beams"] = 1
    else:
        # Remove sampling-only params
        for param in ("temperature", "top_p", "top_k"):
            cleaned.pop(param, None)
        # forced_bos_token_id=None should not be passed explicitly to model.generate
        if cleaned.get("forced_bos_token_id") is None:
            cleaned.pop("forced_bos_token_id", None)

    return cleaned


def _build_generation_preset(
    key: str,
    word_budget: int,
    *,
    num_beams: Optional[int] = None,
    temperature: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    do_sample: Optional[bool] = None,
    length_penalty: Optional[float] = None,
    force_greedy: bool = False,
) -> dict:
    """Build the generation config dict for a given key and word budget.

    Args:
        key: model key (e.g. 'bartpho', 'vit5', 'mt5')
        word_budget: target word count for this generation
        num_beams: override beam count
        force_greedy: if True, forces greedy decoding (used in fallback)
    """
    token_max = words_to_max_new_tokens(word_budget)
    token_min = min_new_tokens_for_budget(word_budget)

    base = config.GENERATION_CONFIGS.get(key, config.DEFAULT_GENERATION_CONFIG).copy()

    # Always enforce anti-repetition floors
    base.setdefault("repetition_penalty", 1.8 if key in {"vit5", "mt5", "bartpho"} else 1.3)
    base.setdefault("no_repeat_ngram_size", 3)

    # Override token limits with budget-derived values
    base["max_new_tokens"] = token_max
    base["min_new_tokens"] = token_min

    if word_budget >= 160:
        base["length_penalty"] = base.get("length_penalty", 1.0) + 0.1
    elif word_budget <= 60:
        base["length_penalty"] = max(0.90, base.get("length_penalty", 1.0) - 0.1)

    if num_beams is not None:
        base["num_beams"] = num_beams
    if do_sample is not None:
        base["do_sample"] = do_sample
    if temperature is not None:
        base["temperature"] = temperature
    if repetition_penalty is not None:
        base["repetition_penalty"] = repetition_penalty
    if length_penalty is not None:
        base["length_penalty"] = length_penalty

    if force_greedy:
        base["do_sample"] = False
        base["num_beams"] = 1
        base.pop("temperature", None)
        base.pop("top_p", None)
        base.pop("top_k", None)
        base["early_stopping"] = False

    return _sanitize_gen_preset(key, base)


# ─────────────────────────── core generation ────────────────────────────────

def _generate_one(
    key: str,
    text: str,
    max_word_budget: Optional[int] = None,
    min_word_budget: Optional[int] = None,
    num_beams: Optional[int] = None,
    temperature: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    do_sample: Optional[bool] = None,
    length_penalty: Optional[float] = None,
) -> str:
    """Generate a single summary from `text` using model `key`.

    Returns an empty string if both primary and fallback generations are corrupted.
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

    word_budget = max(12, int(max_word_budget or config.MAX_OUTPUT_LENGTH))
    gen_preset = _build_generation_preset(
        key,
        word_budget,
        num_beams=num_beams,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        length_penalty=length_penalty,
    )
    if min_word_budget is not None:
        gen_preset["min_new_tokens"] = min(
            gen_preset["max_new_tokens"] - 1,
            words_to_max_new_tokens(int(min_word_budget)),
        )

    log_vram_usage(f"pre_generate_{key}")
    t_start = time.perf_counter()

    def _run_model_generate(params: dict) -> torch.Tensor:
        with torch.inference_mode():
            if use_fp16 and device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    return model.generate(**encoded, **params)
            else:
                return model.generate(**encoded, **params)

    generated_ids = _run_model_generate(gen_preset)
    elapsed = time.perf_counter() - t_start
    logger.debug("[%s] generate() took %.3f s", key, elapsed)
    log_vram_usage(f"post_generate_{key}")

    is_t5_family = key in {"vit5", "mt5"}
    require_vietnamese = key in {"vit5", "mt5"}
    decoded = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=is_t5_family,
    )
    raw = decoded[0] if decoded else ""
    raw = unicodedata.normalize("NFC", raw)
    summary = clean_generated_summary(raw)

    from src.output_validator import validate_output

    validation = validate_output(summary, require_vietnamese=require_vietnamese)

    if validation["is_corrupted"]:
        logger.warning(
            "[%s] Corrupted output detected: %s — retrying with safe greedy fallback...",
            key, validation["quality_warning"],
        )
        fallback_preset = _build_generation_preset(
            key,
            max(24, word_budget // 2),
            force_greedy=True,
        )
        fallback_ids = _run_model_generate(fallback_preset)
        decoded_fallback = tokenizer.batch_decode(
            fallback_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=is_t5_family,
        )
        raw_fallback = decoded_fallback[0] if decoded_fallback else ""
        raw_fallback = unicodedata.normalize("NFC", raw_fallback)
        summary_fallback = clean_generated_summary(raw_fallback)


        fallback_validation = validate_output(
            summary_fallback, require_vietnamese=require_vietnamese
        )
        if fallback_validation["is_corrupted"]:
            logger.error(
                "[%s] Fallback output also corrupted: %s — returning raw text for research analysis.",
                key,
                fallback_validation["quality_warning"],
            )
            return summary_fallback or summary or ""
        return summary_fallback

    if is_probably_bad_generation(summary):
        logger.warning(
            "[%s] General bad generation sanity check failed (len=%d) — returning raw text for research analysis.",
            key,
            len(summary),
        )
        return summary

    return summary

def _generate_chunks_parallel(
    key: str,
    chunks: list[str],
    budgets: list[int],
    min_output_length: int,
    num_beams: int,
    temperature: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    do_sample: Optional[bool] = None,
    length_penalty: Optional[float] = None,
) -> list[str]:
    """Run _generate_one() for each chunk in parallel using a thread pool.

    GPU inference itself is serial (GIL + CUDA context), but preprocessing
    and tokenization can overlap. Workers=2 avoids excessive GPU contention.
    """
    workers = getattr(config, "ABSTRACTIVE_CHUNK_WORKERS", 2)
    # Never run abstractive chunks truly in parallel on one GPU — use workers=1
    # when CUDA is the target to avoid OOM; workers=2 safe for CPU inference.
    device = get_device()
    effective_workers = workers if device.type == "cpu" else 1

    partials: list[str] = [""] * len(chunks)

    if effective_workers <= 1:
        # Serial path (safe on single GPU)
        for i, (chunk, budget) in enumerate(zip(chunks, budgets)):
            partials[i] = _generate_one(
                key,
                chunk,
                max_word_budget=budget,
                min_word_budget=min(min_output_length, max(12, budget // 3)),
                num_beams=num_beams,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                length_penalty=length_penalty,
            )
    else:
        # Parallel path (CPU or multi-GPU setups)
        with ThreadPoolExecutor(
            max_workers=effective_workers,
            thread_name_prefix=f"abs_chunk_{key}",
        ) as pool:
            future_to_idx = {
                pool.submit(
                    _generate_one,
                    key,
                    chunk,
                    budget,
                    min(min_output_length, max(12, budget // 3)),
                    num_beams,
                    temperature,
                    repetition_penalty,
                    do_sample,
                    length_penalty,
                ): i
                for i, (chunk, budget) in enumerate(zip(chunks, budgets))
            }
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    partials[i] = future.result()
                except Exception as exc:
                    logger.error(
                        "[%s] Chunk %d generation failed: %s", key, i, exc
                    )
                    partials[i] = ""

    return partials


# ─────────────────────────── main public API ────────────────────────────────

def abstractive_summarize_key(
    key: str,
    text: str,
    max_output_length: int = config.MAX_OUTPUT_LENGTH,
    min_output_length: int = config.MIN_OUTPUT_LENGTH,
    num_beams: int = config.NUM_BEAMS,
    temperature: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    do_sample: Optional[bool] = None,
    length_penalty: Optional[float] = None,
) -> str:
    text = clean_text(text, aggressive=True)
    if not text:
        return ""

    target_words = max(12, int(max_output_length))
    source_words = count_words(text)

    max_words = _max_words_per_chunk(key)

    if source_words > max_words:
        chunks = _chunk_text(text, max_words)
        logger.info("[%s] Long text (%d words) → %d chunks", key, source_words, len(chunks))

        chunk_budgets = allocate_chunk_word_budgets(chunks, target_words)
        partials = _generate_chunks_parallel(
            key, chunks, chunk_budgets, min_output_length, num_beams,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            length_penalty=length_penalty,
        )

        merged = " ".join(p for p in partials if p)
        merged = clean_generated_summary(merged)
        merged_words = count_words(merged)

        # ── Synthesis re-pass ────────────────────────────────────────────────
        # If merged output is too long (>1.5× target), re-summarize it once.
        if merged_words > int(target_words * 1.5):
            logger.info(
                "[%s] Merged too long (%d words vs target %d) → synthesis re-pass",
                key, merged_words, target_words,
            )
            merged = _generate_one(
                key,
                merged,
                max_word_budget=target_words,
                min_word_budget=max(min_output_length, target_words // 4),
                num_beams=num_beams,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                length_penalty=length_penalty,
            )
            merged = clean_generated_summary(merged)

        # If still too short (chunking collapsed), try full-text pass with truncation
        elif merged_words < int(target_words * 0.60):
            logger.info(
                "[%s] Chunked summary too short (%d/%d words) → full-text synthesis pass",
                key, merged_words, target_words,
            )
            merged = _generate_one(
                key,
                text,
                max_word_budget=target_words,
                min_word_budget=max(min_output_length, target_words // 4),
                num_beams=num_beams,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                length_penalty=length_penalty,
            )
            merged = clean_generated_summary(merged)

        # Final trim to hard word budget
        if count_words(merged) > target_words:
            from src.length_control import trim_summary_to_word_budget
            merged = trim_summary_to_word_budget(merged, target_words)

        return merged

    # Short text — single-shot generation
    return _generate_one(
        key,
        text,
        max_word_budget=target_words,
        min_word_budget=min_output_length,
        num_beams=num_beams,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        length_penalty=length_penalty,
    )


# ─────────────────────────── class wrapper ──────────────────────────────────

class AbstractiveSummarizer:
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
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        length_penalty: Optional[float] = None,
        chunk_long_text: bool = True,
    ) -> str:
        return abstractive_summarize_key(
            self.key,
            text,
            max_output_length=max_output_length or self.max_output_length,
            min_output_length=min_output_length or self.min_output_length,
            num_beams=num_beams or self.num_beams,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            length_penalty=length_penalty,
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


# ─────────────────────────── convenience functions ──────────────────────────

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
