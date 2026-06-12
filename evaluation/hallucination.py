"""Hallucination and factual consistency auditing."""

from __future__ import annotations

from typing import Any

from evaluation.readability import readability_scores
from src.fact_check import check_consistency
from src.preprocess import split_sentences
from utils.metrics import lexical_overlap


def _semantic_sentence_scores(summary_sentences: list[str], source_sentences: list[str]) -> list[float]:
    """Lightweight semantic alignment without heavy NLI models."""
    if not summary_sentences or not source_sentences:
        return []
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        src_emb = model.encode(source_sentences, normalize_embeddings=True)
        sum_emb = model.encode(summary_sentences, normalize_embeddings=True)
        scores = (sum_emb @ src_emb.T).max(axis=1)
        return [round(float(s), 4) for s in scores]
    except Exception:
        return [
            max((lexical_overlap(sent, src) for src in source_sentences), default=0.0)
            for sent in summary_sentences
        ]


def audit_summary(
    summary: str,
    source_text: str,
    *,
    chunks: list[dict[str, Any]] | None = None,
    mode: str = "fast",
) -> dict[str, Any]:
    """Combine semantic consistency with lexical source verification."""
    consistency = check_consistency(summary, source_text, mode=mode)  # type: ignore[arg-type]
    summary_sentences = split_sentences(summary)
    source_sentences = split_sentences(source_text)
    semantic_scores = _semantic_sentence_scores(summary_sentences, source_sentences)
    sentence_audits = [
        {
            "sentence": sent,
            "semantic_alignment": semantic_scores[idx] if idx < len(semantic_scores) else 0.0,
            "lexical_alignment": max((lexical_overlap(sent, src) for src in source_sentences), default=0.0),
        }
        for idx, sent in enumerate(summary_sentences)
    ]
    chunk_hits = []
    if chunks:
        for sentence in summary_sentences:
            if len(sentence) < 8:
                continue
            best = max(
                (
                    {
                        "chunk_id": c.get("chunk_id"),
                        "score": lexical_overlap(sentence, c.get("text", "")),
                    }
                    for c in chunks
                ),
                key=lambda x: x["score"],
                default={"chunk_id": None, "score": 0.0},
            )
            chunk_hits.append(best)

    supported = sum(1 for h in chunk_hits if h["score"] >= 0.35)
    total = max(1, len(chunk_hits))
    grounding_coverage = round(supported / total, 4) if chunk_hits else 0.0

    sem_supported = sum(1 for row in sentence_audits if row["semantic_alignment"] >= 0.55)
    sem_total = max(1, len(sentence_audits))
    semantic_coverage = round(sem_supported / sem_total, 4)

    # Calculate faithfulness and coverage scores
    from evaluation.metrics import compute_faithfulness_score, compute_coverage_score, compute_info_retention

    faithfulness_val = compute_faithfulness_score(summary, source_text)
    coverage_val = compute_coverage_score(summary, source_text)
    
    # Estimate compression ratio for info retention
    pred_words = len(summary.split())
    src_words = len(source_text.split())
    comp_ratio = pred_words / max(1, src_words)
    # Use lexical overlap with source as a proxy for ROUGE-L in info_retention
    source_overlap = lexical_overlap(source_text, summary)
    info_retention_val = compute_info_retention(source_overlap, comp_ratio)

    status = consistency.get("status", "unknown")
    nli_available = bool(semantic_scores)
    if grounding_coverage >= 0.7 and consistency.get("consistency_score", 0) >= 0.55 and semantic_coverage >= 0.6:
        risk = "low"
    elif grounding_coverage >= 0.4 or semantic_coverage >= 0.45:
        risk = "medium"
    else:
        risk = "high"

    return {
        **consistency,
        "grounding_coverage": grounding_coverage,
        "semantic_coverage": semantic_coverage,
        "chunk_verification": chunk_hits[:20],
        "sentence_audits": sentence_audits[:30],
        "readability": readability_scores(summary),
        "hallucination_risk": risk,
        "contradiction_flag": status in {"unsupported", "suspicious"},
        "nli_mode": "embedding-alignment" if nli_available else "lexical-fallback",
        "faithfulness_score": faithfulness_val,
        "coverage_score": coverage_val,
        "info_retention": info_retention_val,
    }

