"""
adaptive_context_builder.py — Adaptive Context Builder cho ChatRAG pipeline.

Thay thế/nâng cấp Context Compression với:
  1. Query-aware Hybrid Summary (intent analysis trước)
  2. Dynamic Original Chunk Selection (rerank score ratio)
  3. Adaptive Compression Controller (tiers theo độ dài)
  4. Citation Preservation
  5. Fact Preservation (số, ngày, công thức, tên)
  6. Prompt Composer
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable

from .context_compression import (
    CompressedContext,
    compute_total_context_chars,
    pick_best_hybrid_key,
    _chunk_score,
)
from .perf import StageTimer
from .rag_config import (
    RAG_ADAPTIVE_CONTEXT,
    RAG_DYNAMIC_CHUNK_RATIO,
    RAG_HEAVY_COMPRESSION,
    RAG_LIGHT_COMPRESSION,
    RAG_MEDIUM_COMPRESSION,
    RAG_MIN_RERANK_SCORE,
)

logger = logging.getLogger(__name__)

# ─── Fact extraction patterns ────────────────────────────────────────────────
_FACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\s*(?:%|phần trăm|tỷ|triệu|nghìn|USD|VND|đồng)?\b", re.I),
    re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b"),
    re.compile(r"\b(?:năm|tháng|ngày)\s+\d{1,4}\b", re.I),
    re.compile(r"\b\d+\s*[×x*]\s*\d+\b"),
    re.compile(r"\b[A-Z][a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]+(?:\s+[A-Z][a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]+){1,3}\b"),
]

_COMPRESSION_TIERS = {
    "none": {"min_chars": 0, "max_chars": RAG_LIGHT_COMPRESSION, "ratio": 1.0, "max_tokens": 0},
    "light": {"min_chars": RAG_LIGHT_COMPRESSION, "max_chars": RAG_MEDIUM_COMPRESSION, "ratio": 0.50, "max_tokens": 200},
    "medium": {"min_chars": RAG_MEDIUM_COMPRESSION, "max_chars": RAG_HEAVY_COMPRESSION, "ratio": 0.35, "max_tokens": 280},
    "aggressive": {"min_chars": RAG_HEAVY_COMPRESSION, "max_chars": float("inf"), "ratio": 0.25, "max_tokens": 350},
}


def _estimate_tokens(text: str) -> int:
    """Ước lượng token từ ký tự (tiếng Việt ~3.5 char/token)."""
    return max(1, int(len(text) / 3.5))


def analyze_query_focus(query: str, document_ids: list[str] | None = None) -> tuple[str, str]:
    """
    Phân tích intent + focus từ query — rule-based, không gọi LLM.
    Returns: (intent, focus_phrase)
    """
    from .agent import classify_intent

    intent = classify_intent(query, document_ids)
    query_clean = query.strip()

    focus_keywords: list[str] = []
    number_q = re.search(r"\b(?:bao nhiêu|số lượng|tỷ lệ|phần trăm|kết quả)\b", query_clean, re.I)
    date_q = re.search(r"\b(?:khi nào|thời gian|ngày|năm|tháng)\b", query_clean, re.I)
    compare_q = re.search(r"\b(?:so sánh|khác biệt|đối chiếu)\b", query_clean, re.I)
    method_q = re.search(r"\b(?:phương pháp|thuật toán|kỹ thuật|cách thức)\b", query_clean, re.I)

    if number_q:
        focus_keywords.append("số liệu và thống kê")
    if date_q:
        focus_keywords.append("mốc thời gian và ngày tháng")
    if compare_q:
        focus_keywords.append("so sánh và đối chiếu")
    if method_q:
        focus_keywords.append("phương pháp và kỹ thuật")

    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', query_clean)
    for a, b in quoted:
        term = (a or b).strip()
        if term:
            focus_keywords.append(term)

    if not focus_keywords:
        words = re.findall(r"\w+", query_clean.lower())
        stop = {"là", "gì", "của", "trong", "và", "các", "những", "cho", "về", "theo", "nào", "được", "có", "không"}
        content_words = [w for w in words if w not in stop and len(w) > 2][:5]
        if content_words:
            focus_keywords.append(", ".join(content_words))

    focus = "; ".join(focus_keywords) if focus_keywords else query_clean[:120]
    return intent, focus


def resolve_compression_tier(total_chars: int) -> str:
    if total_chars < RAG_LIGHT_COMPRESSION:
        return "none"
    if total_chars < RAG_MEDIUM_COMPRESSION:
        return "light"
    if total_chars < RAG_HEAVY_COMPRESSION:
        return "medium"
    return "aggressive"


def select_dynamic_chunks(
    chunks: list[dict[str, Any]],
    *,
    ratio: float = RAG_DYNAMIC_CHUNK_RATIO,
    min_score: float = RAG_MIN_RERANK_SCORE,
) -> list[dict[str, Any]]:
    """
    Chọn chunks động: score >= ratio * max_score AND score >= min_score.
    Luôn giữ ít nhất 1 chunk nếu có dữ liệu.
    """
    if not chunks:
        return []

    scored = [(c, _chunk_score(c)) for c in chunks]
    max_score = max(s for _, s in scored)
    threshold = max(min_score, ratio * max_score)

    selected = [c for c, s in scored if s >= threshold]
    if not selected:
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        selected = [ranked[0][0]]
    return sorted(selected, key=_chunk_score, reverse=True)


def extract_facts_from_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trích xuất facts (số, ngày, công thức, tên) từ chunks."""
    facts: list[dict[str, Any]] = []
    seen: set[str] = set()

    for chunk in chunks:
        text = chunk.get("text", "")
        chunk_id = chunk.get("chunk_id") or chunk.get("id", "?")
        for pattern in _FACT_PATTERNS:
            for match in pattern.finditer(text):
                fact_text = match.group(0).strip()
                if len(fact_text) < 2 or fact_text in seen:
                    continue
                seen.add(fact_text)
                facts.append({
                    "text": fact_text,
                    "chunk_id": chunk_id,
                    "filename": chunk.get("filename"),
                    "page": chunk.get("page"),
                    "document_id": chunk.get("document_id"),
                })
    return facts


def merge_fact_chunks(
    selected: list[dict[str, Any]],
    all_chunks: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Đảm bảo chunks chứa facts quan trọng được giữ trong original passages."""
    selected_ids = {c.get("chunk_id") or c.get("id") for c in selected}
    fact_chunk_ids = {f["chunk_id"] for f in facts if f.get("chunk_id")}

    for chunk in all_chunks:
        cid = chunk.get("chunk_id") or chunk.get("id")
        if cid in fact_chunk_ids and cid not in selected_ids:
            selected.append(chunk)
            selected_ids.add(cid)

    return sorted(selected, key=_chunk_score, reverse=True)


def attach_citations_to_summary(
    summary: str,
    chunks: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Gắn metadata citation vào summary — mỗi câu tham chiếu chunk nguồn phù hợp nhất.
    """
    if not summary or not chunks:
        return summary, []

    sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
    citations: list[dict[str, Any]] = []
    annotated: list[str] = []

    for sent_idx, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
        best_chunk = None
        best_overlap = 0
        sent_words = set(re.findall(r"\w+", sentence.lower()))

        for chunk in chunks:
            chunk_words = set(re.findall(r"\w+", chunk.get("text", "").lower()))
            overlap = len(sent_words & chunk_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_chunk = chunk

        if best_chunk:
            cid = best_chunk.get("chunk_id") or best_chunk.get("id", "?")
            page = best_chunk.get("page")
            doc_id = best_chunk.get("document_id", "")
            cite_tag = f"[chunk:{cid}" + (f", p.{page}" if page else "") + "]"
            annotated.append(f"{sentence} {cite_tag}")
            citations.append({
                "sentence_index": sent_idx,
                "chunk_id": cid,
                "document_id": doc_id,
                "page": page,
                "filename": best_chunk.get("filename"),
            })
        else:
            annotated.append(sentence)

    return " ".join(annotated), citations


def _generate_query_aware_summary(
    chunks: list[dict[str, Any]],
    query: str,
    focus: str,
    *,
    compression_ratio: float,
    max_target_tokens: int,
) -> tuple[str, str | None, str | None]:
    """Sinh Hybrid Summary query-aware — prepend focus vào input."""
    from pipeline.hybrid_summarizer import summarize_retrieved_chunks

    query_prefix = (
        f"[TRỌNG TÂM CÂU HỎI: {focus}]\n"
        f"[CÂU HỎI: {query}]\n\n"
    )
    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        c = dict(chunk)
        if i == 0:
            c["text"] = query_prefix + c.get("text", "")
        enriched_chunks.append(c)

    return summarize_retrieved_chunks(
        enriched_chunks,
        query=query,
        compression_ratio=compression_ratio,
        max_target_tokens=max_target_tokens,
    )


def build_adaptive_context(
    retrieved: list[dict[str, Any]],
    query: str,
    timer: StageTimer | None = None,
    *,
    document_ids: list[str] | None = None,
    stage_callback: Callable[[str, str], None] | None = None,
) -> CompressedContext:
    """
    Adaptive Context Builder — entry point chính.
    Trả về CompressedContext mở rộng với context_details.
    """
    def _stage(name: str, status: str = "active") -> None:
        if stage_callback:
            stage_callback(name, status)

    if not retrieved:
        return CompressedContext(
            enabled=False,
            skipped_reason="empty",
            all_retrieved=[],
            mode="adaptive",
        )

    if not RAG_ADAPTIVE_CONTEXT:
        from .context_compression import compress_retrieved_context
        return compress_retrieved_context(retrieved, query, timer)

    from .cache import (
        adaptive_context_cache_key,
        get_cached_adaptive_context,
        set_cached_adaptive_context,
    )

    cache_key = adaptive_context_cache_key(query, retrieved)
    cached = get_cached_adaptive_context(cache_key)
    if cached is not None:
        logger.debug("Adaptive context cache hit for query=%s", query[:40])
        return cached

    t0 = time.perf_counter()
    input_chars = compute_total_context_chars(retrieved)
    input_tokens_est = _estimate_tokens("".join(c.get("text", "") for c in retrieved))

    _stage("acb_intent", "active")
    if timer:
        timer.start("acb_intent")
    intent, focus = analyze_query_focus(query, document_ids)
    if timer:
        timer.stop("acb_intent")
    _stage("acb_intent", "done")

    tier_name = resolve_compression_tier(input_chars)
    tier_cfg = _COMPRESSION_TIERS[tier_name]

    _stage("acb_chunks", "active")
    if timer:
        timer.start("acb_chunks")
    dynamic_chunks = select_dynamic_chunks(retrieved)
    facts = extract_facts_from_chunks(retrieved)
    dynamic_chunks = merge_fact_chunks(dynamic_chunks, retrieved, facts)
    if timer:
        timer.stop("acb_chunks")
    _stage("acb_chunks", "done")

    summary = ""
    model_used: str | None = None
    algo_key: str | None = None
    citations: list[dict[str, Any]] = []

    if tier_name != "none":
        _stage("acb_summary", "active")
        if timer:
            timer.start("acb_summary")
        hybrid_key, _, _ = pick_best_hybrid_key()
        try:
            summary, model_used, algo_key = _generate_query_aware_summary(
                retrieved,
                query,
                focus,
                compression_ratio=tier_cfg["ratio"],
                max_target_tokens=tier_cfg["max_tokens"],
            )
        except Exception as exc:
            logger.error("Query-aware hybrid summary failed: %s", exc, exc_info=True)
            summary, model_used, algo_key = "", None, hybrid_key
        if timer:
            timer.stop("acb_summary")
        _stage("acb_summary", "done")

        if summary and len(summary.split()) >= 5:
            _stage("acb_facts", "active")
            summary, citations = attach_citations_to_summary(summary, retrieved)
            _stage("acb_facts", "done")
        else:
            logger.warning("Adaptive summary rỗng/ngắn — fallback dynamic chunks only")
            summary = ""
            tier_name = "none"

    enabled = True
    skipped_reason: str | None = None

    if tier_name == "none":
        skipped_reason = "short_context"
        top_chunks = dynamic_chunks or retrieved
        summary = ""
    elif summary:
        top_chunks = dynamic_chunks
    else:
        skipped_reason = "summary_failed"
        top_chunks = dynamic_chunks or retrieved
        enabled = bool(top_chunks)

    output_chars = len(summary) + compute_total_context_chars(top_chunks)
    output_tokens_est = _estimate_tokens(summary) + _estimate_tokens(
        "\n".join(c.get("text", "") for c in top_chunks)
    )
    compression_ratio = round(output_chars / max(input_chars, 1), 4)
    elapsed = time.perf_counter() - t0

    token_reduction = round(1.0 - output_tokens_est / max(input_tokens_est, 1), 4)
    latency_saving_est = round(max(0.0, (input_tokens_est - output_tokens_est) * 0.002), 4)

    context_details = {
        "mode": "adaptive",
        "query_intent": intent,
        "query_focus": focus,
        "compression_tier": tier_name,
        "input_chars": input_chars,
        "output_chars": output_chars,
        "input_tokens_est": input_tokens_est,
        "output_tokens_est": output_tokens_est,
        "summary_tokens": _estimate_tokens(summary),
        "token_reduction": token_reduction,
        "compression_ratio": compression_ratio,
        "chunks_kept": len(top_chunks),
        "chunks_total": len(retrieved),
        "facts_preserved": len(facts),
        "citations_count": len(citations),
        "dynamic_threshold": round(RAG_DYNAMIC_CHUNK_RATIO * max(_chunk_score(c) for c in retrieved), 4),
        "latency_saving_estimate_s": latency_saving_est,
    }

    _stage("acb_compose", "active")
    _stage("acb_compose", "done")

    logger.info(
        "🧠 Adaptive Context: tier=%s, %d→%d chars (%.1f%%), chunks=%d/%d, facts=%d, intent=%s",
        tier_name,
        input_chars,
        output_chars,
        100.0 * compression_ratio,
        len(top_chunks),
        len(retrieved),
        len(facts),
        intent,
    )

    result = CompressedContext(
        enabled=enabled,
        skipped_reason=skipped_reason,
        hybrid_summary=summary,
        top_original_chunks=top_chunks,
        all_retrieved=retrieved,
        compression_ratio=compression_ratio,
        input_chars=input_chars,
        output_chars=output_chars,
        model_used=model_used,
        hybrid_algo_key=algo_key,
        latency_s=elapsed,
        mode="adaptive",
        query_intent=intent,
        query_focus=focus,
        compression_tier=tier_name,
        dynamic_chunks_kept=len(top_chunks),
        summary_tokens=_estimate_tokens(summary),
        input_tokens_est=input_tokens_est,
        output_tokens_est=output_tokens_est,
        facts_preserved_count=len(facts),
        citations=citations,
        context_details=context_details,
        latency_saving_estimate=latency_saving_est,
    )

    set_cached_adaptive_context(cache_key, result)
    return result
