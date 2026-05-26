"""Map-reduce abstractive summarization for long Vietnamese documents."""

from __future__ import annotations

import time
from typing import Any

from src.abstractive import abstractive_summarize_key
from src.extractive import summarize_extractive_algorithm
from src.preprocess import clean_text, split_sentences
from src.utils import count_words, logger


def hierarchical_summarize(
    source_text: str,
    chunks: list[dict[str, Any]],
    *,
    model_key: str = "vit5",
    max_chunk_output: int = 80,
    max_global_output: int = 180,
    use_extractive_map: bool = False,
) -> dict[str, Any]:
    """Summarize chunk → section → global with optional abstractive reduce steps."""
    start = time.perf_counter()
    chunk_summaries: list[dict[str, Any]] = []

    for chunk in chunks[:48]:
        text = (chunk.get("text") or "").strip()
        if len(text.split()) < 12:
            continue
        if use_extractive_map:
            summary = summarize_extractive_algorithm(text, "textrank", sentence_count=2)["summary"]
            method = "extractive-textrank"
        else:
            try:
                summary = abstractive_summarize_key(text, model_key, max_output_length=max_chunk_output)
                method = f"abstractive-{model_key}"
            except Exception as exc:
                logger.warning("Chunk abstractive failed, TextRank fallback: %s", exc)
                summary = summarize_extractive_algorithm(text, "textrank", sentence_count=2)["summary"]
                method = "extractive-fallback"

        chunk_summaries.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "section_path": chunk.get("section_path", []),
                "summary": summary,
                "method": method,
                "word_count": count_words(summary),
            }
        )

    section_map: dict[str, list[str]] = {}
    for item in chunk_summaries:
        section = " / ".join(item.get("section_path") or ["Unsectioned"])
        section_map.setdefault(section, []).append(item["summary"])

    section_summaries = [
        {"section": section, "summary": clean_text(" ".join(parts))}
        for section, parts in section_map.items()
    ]

    reduce_input = "\n".join(s["summary"] for s in section_summaries) or source_text[:12000]
    try:
        global_summary = abstractive_summarize_key(
            reduce_input,
            model_key,
            max_output_length=max_global_output,
        )
        global_method = f"abstractive-{model_key}"
    except Exception as exc:
        logger.warning("Global abstractive failed: %s", exc)
        global_summary = summarize_extractive_algorithm(reduce_input, "textrank", sentence_count=6)["summary"]
        global_method = "extractive-textrank"

    return {
        "strategy": "map-reduce",
        "model_key": model_key,
        "chunk_summaries": chunk_summaries,
        "section_summaries": section_summaries,
        "global_summary": global_summary,
        "global_method": global_method,
        "source_words": count_words(source_text),
        "output_words": count_words(global_summary),
        "elapsed_seconds": round(time.perf_counter() - start, 4),
    }
