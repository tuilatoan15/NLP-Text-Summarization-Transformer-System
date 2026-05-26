"""Vietnamese sentence segmentation and token accounting."""

from __future__ import annotations

import os
import re
from functools import lru_cache

from utils.logger import logger


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
WORD_RE = re.compile(r"[\wÀ-ỹĐđ]+|[^\w\s]", re.UNICODE)


class VietnameseTokenizer:
    """Tokenization facade with underthesea/pyvi and tiktoken/transformers fallbacks."""

    def __init__(self, token_model_name: str | None = None, use_vietnamese_segmenter: bool = False) -> None:
        self.token_model_name = token_model_name
        self.use_vietnamese_segmenter = use_vietnamese_segmenter or os.getenv("INGEST_USE_VI_SEGMENTER") == "1"
        self._hf_tokenizer = None
        self._tiktoken_encoding = None

    def split_sentences(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        if self.use_vietnamese_segmenter:
            try:
                from underthesea import sent_tokenize

                sentences = [sentence.strip() for sentence in sent_tokenize(text) if sentence and sentence.strip()]
                if sentences:
                    return sentences
            except Exception as exc:
                logger.debug("underthesea sent_tokenize fallback: %s", exc)

            try:
                from pyvi import ViTokenizer

                normalized = ViTokenizer.tokenize(text).replace("_", " ")
                sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(normalized) if sentence.strip()]
                if sentences:
                    return sentences
            except Exception as exc:
                logger.debug("pyvi sentence split fallback: %s", exc)

        return [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(text) if sentence.strip()]

    def word_tokenize(self, text: str) -> list[str]:
        text = text or ""
        if self.use_vietnamese_segmenter:
            try:
                from underthesea import word_tokenize

                tokens = word_tokenize(text, format="text").split()
                if tokens:
                    return tokens
            except Exception:
                pass
        return [match.group(0) for match in WORD_RE.finditer(text)]

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        tokenizer = self._load_transformers_tokenizer()
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass

        encoding = self._load_tiktoken()
        if encoding is not None:
            try:
                return len(encoding.encode(text))
            except Exception:
                pass

        # Vietnamese syllable-level whitespace tokenization underestimates subword
        # tokenizers, so use a mild multiplier for safer chunk sizes.
        return max(1, int(len(self.word_tokenize(text)) * 1.25))

    def trim_to_tokens(self, text: str, max_tokens: int) -> str:
        sentences = self.split_sentences(text)
        kept: list[str] = []
        total = 0
        for sentence in sentences:
            count = self.count_tokens(sentence)
            if kept and total + count > max_tokens:
                break
            kept.append(sentence)
            total += count
        return " ".join(kept).strip()

    def _load_transformers_tokenizer(self):
        if not self.token_model_name:
            return None
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer
        try:
            from transformers import AutoTokenizer

            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.token_model_name, use_fast=True)
            return self._hf_tokenizer
        except Exception as exc:
            logger.debug("transformers tokenizer unavailable for %s: %s", self.token_model_name, exc)
            self._hf_tokenizer = None
            return None

    def _load_tiktoken(self):
        if self._tiktoken_encoding is not None:
            return self._tiktoken_encoding
        try:
            import tiktoken

            self._tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
            return self._tiktoken_encoding
        except Exception as exc:
            logger.debug("tiktoken unavailable: %s", exc)
            self._tiktoken_encoding = None
            return None


@lru_cache(maxsize=8)
def cached_token_count(text: str, token_model_name: str | None = None) -> int:
    return VietnameseTokenizer(token_model_name=token_model_name).count_tokens(text)
