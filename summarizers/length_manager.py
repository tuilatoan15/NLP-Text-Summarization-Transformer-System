"""
SummaryLengthManager — Automates target length settings and hierarchical summarization for long documents.
"""

from __future__ import annotations

from typing import Any, Tuple
from src.preprocess import clean_text, split_sentences
from src.utils import count_words, logger


class SummaryLengthManager:
    """
    Analyzes input text and determines optimal summary length configurations
    for both extractive and abstractive models.
    """

    @staticmethod
    def analyze_input(text: str) -> dict[str, Any]:
        """
        Analyzes the input text for word count, sentence count, paragraph count.
        """
        cleaned_text = clean_text(text or "", aggressive=True)
        words = cleaned_text.split()
        word_count = len(words)

        sentences = split_sentences(cleaned_text)
        sentence_count = len(sentences)

        paragraphs = [p for p in text.split("\n") if p.strip()]
        paragraph_count = len(paragraphs)

        # Categorize input length
        if word_count < 500:
            suggested_mode = "short"
        elif word_count <= 3000:
            suggested_mode = "standard"
        else:
            suggested_mode = "detailed"

        is_extremely_long = word_count > 10000

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "suggested_mode": suggested_mode,
            "is_extremely_long": is_extremely_long,
        }

    @classmethod
    def get_extractive_sentences(cls, length_mode: str, analysis: dict[str, Any]) -> int:
        """
        Returns the number of sentences to extract based on the length mode.
        """
        mode = length_mode.lower().strip() if length_mode else "auto"
        if mode == "auto":
            mode = analysis.get("suggested_mode", "standard")

        if mode == "short":
            return 3
        elif mode == "standard":
            return 5
        elif mode == "detailed" or mode == "extremely_long":
            return 8
        return 5

    @classmethod
    def get_abstractive_limits(
        cls, model_key: str, length_mode: str, analysis: dict[str, Any]
    ) -> Tuple[int, int]:
        """
        Returns (min_new_tokens, max_new_tokens) for abstractive generation.
        """
        del model_key
        mode = length_mode.lower().strip() if length_mode else "auto"
        if mode == "auto":
            mode = analysis.get("suggested_mode", "standard")

        if mode == "short":
            return 30, 100
        elif mode == "standard":
            return 60, 200
        elif mode == "detailed" or mode == "extremely_long":
            return 120, 400
        return 60, 200

    @classmethod
    def hierarchical_summarize_pipeline(
        cls, text: str, algorithm: str, length_mode: str, group: str
    ) -> str:
        """
        Orchestrates Chunk -> Summarize each -> Merge -> Final Summary pipeline
        for extremely long documents (>10,000 words).
        """
        cleaned = clean_text(text, aggressive=True)
        sentences = split_sentences(cleaned)
        total_words = count_words(cleaned)

        # Split text into chunks of max 1500 words
        max_chunk_words = 1500
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_words = 0

        for sent in sentences:
            sent_words = count_words(sent)
            if current_chunk and current_words + sent_words > max_chunk_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_words = 0
            current_chunk.append(sent)
            current_words += sent_words

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        logger.info(
            "SummaryLengthManager: Running hierarchical pipeline for %s (%s, %d words) with %d chunks",
            algorithm,
            group,
            total_words,
            len(chunks),
        )

        if group == "extractive":
            from src.extractive import summarize_extractive_algorithm

            chunk_summaries = []
            for chunk in chunks:
                # Extract 3 sentences from each chunk
                res = summarize_extractive_algorithm(chunk, algorithm, sentence_count=3)
                chunk_summaries.append(res.get("summary", ""))

            merged_text = " ".join(chunk_summaries)
            analysis = {"suggested_mode": "detailed"}
            final_sentences = cls.get_extractive_sentences(length_mode, analysis)
            final_res = summarize_extractive_algorithm(merged_text, algorithm, sentence_count=final_sentences)
            return final_res.get("summary", "")

        elif group == "abstractive":
            from src.abstractive import abstractive_summarize_key

            chunk_summaries = []
            for chunk in chunks:
                # Summarize chunk with short preset
                summary = abstractive_summarize_key(chunk, algorithm, max_output_length=100, min_output_length=30)
                chunk_summaries.append(summary)

            merged_text = " ".join(chunk_summaries)
            analysis = {"suggested_mode": "detailed"}
            min_tok, max_tok = cls.get_abstractive_limits(algorithm, length_mode, analysis)
            final_summary = abstractive_summarize_key(
                merged_text,
                algorithm,
                max_output_length=max_tok,
                min_output_length=min_tok,
            )
            return final_summary

        return ""
