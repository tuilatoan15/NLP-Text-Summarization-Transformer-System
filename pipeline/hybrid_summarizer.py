"""
hybrid_summarizer.py — Pipeline tóm tắt nhiều tầng (Hybrid Summarization Pipeline).
Thực hiện nén văn bản bằng thuật toán Extractive trước (loại bỏ thông tin rác),
sau đó đưa qua mô hình sinh Abstractive (ViT5 / BARTPho) để tạo bản tóm tắt tự nhiên.
Giải quyết triệt để lỗi tràn bộ nhớ (VRAM OOM) và lỗi cắt cụt văn bản (truncation) của Transformer.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, List

from summarizers.extractive.extractive_summarizer import EXTRACTIVE_RUNNERS
from summarizers.abstractive.abstractive_summarizer import AbstractiveSummarizer
from src.preprocess import clean_text, split_sentences
from src.utils import count_words

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Split text into semantically cohesive chunks using sentence embeddings."""
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", threshold: float = 0.5):
        self.model_name = model_name
        self.threshold = threshold
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def chunk_document(self, sentences: list[str]) -> list[str]:
        if len(sentences) <= 1:
            return sentences
        
        try:
            model = self._get_model()
            embeddings = model.encode(sentences, normalize_embeddings=True)
            
            chunks = []
            current_chunk = [sentences[0]]
            
            for i in range(len(sentences) - 1):
                # Calculate cosine similarity between adjacent sentence embeddings
                sim = float(embeddings[i] @ embeddings[i+1])
                if sim < self.threshold:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentences[i+1]]
                else:
                    current_chunk.append(sentences[i+1])
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            return chunks
        except Exception as exc:
            logger.warning(f"Semantic chunking failed, fallback to sentence grouping: {exc}")
            chunks = []
            for i in range(0, len(sentences), 3):
                chunks.append(" ".join(sentences[i:i+3]))
            return chunks


class HybridSummarizer:
    """
    Hệ thống tóm tắt thông minh nhiều tầng kết hợp Extractive + Abstractive.
    """

    def __init__(self, abstractive_model_key: str = "vit5") -> None:
        self.abstractive_model_key = abstractive_model_key
        self.abstractive_engine = AbstractiveSummarizer(model_name=abstractive_model_key)

    def build_condensed_context(
        self,
        text: str,
        extractive_algo: str = "textrank",
        compression_ratio: float = 0.35,
        use_semantic_chunking: bool = False,
    ) -> str:
        """Chạy stage extractive một lần — dùng chung cho nhóm hybrid cùng backbone."""
        cleaned_text = clean_text(text, aggressive=True)
        sentences = split_sentences(cleaned_text)
        input_word_count = count_words(cleaned_text)
        if input_word_count < 10 or len(sentences) <= 3:
            return cleaned_text

        num_sentences = max(3, min(int(len(sentences) * compression_ratio), 25))
        if use_semantic_chunking:
            chunker = SemanticChunker(threshold=0.45)
            semantic_chunks = chunker.chunk_document(sentences)
            selected_sentences: list[str] = []
            sents_per_chunk = max(1, int(num_sentences / max(1, len(semantic_chunks))))
            for chunk in semantic_chunks:
                chunk_sents = split_sentences(chunk)
                if len(chunk_sents) <= sents_per_chunk:
                    selected_sentences.extend(chunk_sents)
                else:
                    extractive_runner = EXTRACTIVE_RUNNERS.get(extractive_algo) or EXTRACTIVE_RUNNERS["textrank"]
                    details = extractive_runner(chunk, sentence_count=sents_per_chunk)
                    selected_sentences.extend(split_sentences(details.get("summary", "")))
            return " ".join(selected_sentences[:num_sentences])

        extractive_runner = EXTRACTIVE_RUNNERS.get(extractive_algo) or EXTRACTIVE_RUNNERS["textrank"]
        condensed_details = extractive_runner(cleaned_text, sentence_count=num_sentences)
        return condensed_details.get("summary", "")

    def summarize_from_condensed(
        self,
        condensed_text: str,
        max_target_tokens: int = 200,
        temperature: float = 0.7,
        num_beams: int = 4,
        repetition_penalty: float = 2.0,
    ) -> str:
        """Sinh abstractive từ condensed context đã có — bỏ qua stage extractive."""
        if not (condensed_text or "").strip():
            return ""
        return self._run_abstractive_direct(
            condensed_text,
            max_target_tokens,
            temperature,
            num_beams,
            repetition_penalty,
        )

    def summarize(
        self,
        text: str,
        compression_ratio: float = 0.35,
        max_target_tokens: int = 200,
        extractive_algo: str = "textrank",
        temperature: float = 0.7,
        num_beams: int = 4,
        repetition_penalty: float = 2.0,
        use_semantic_chunking: bool = False
    ) -> str:
        """
        Thực hiện tóm tắt văn bản thông minh:
          1. Tách câu và lọc tiền xử lý.
          2. Dùng thuật toán Extractive (hoặc Semantic Chunking) để lọc ra Top N% câu quan trọng nhất.
          3. Ghép các câu cốt lõi lại thành văn bản cô đọng (Condensed Context).
          4. Đưa văn bản cô đọng vào Transformer (Abstractive) để sinh bản tóm tắt tối ưu.

        Args:
            text:                     Văn bản gốc đầu vào
            compression_ratio:        Tỷ lệ nén câu cho bước Extractive (0.1–1.0)
            max_target_tokens:        Độ dài tối đa của bản tóm tắt sinh ra
            extractive_algo:          Thuật toán lọc câu (textrank | lexrank | lsa)
            temperature:              Nhiệt độ sinh từ của Transformer
            num_beams:                Số lượng beam search
            repetition_penalty:       Tham số phạt lặp từ
            use_semantic_chunking:    Sử dụng Semantic Chunking để lọc câu ngữ nghĩa

        Returns:
            Bản tóm tắt cuối cùng dạng sinh (Abstractive Summary)
        """
        t_start = time.perf_counter()
        cleaned_text = clean_text(text, aggressive=True)
        input_word_count = count_words(cleaned_text)
        sentences = split_sentences(cleaned_text)
        if input_word_count < 10 or len(sentences) <= 3:
            logger.info("Văn bản quá ngắn, chuyển trực tiếp sang tóm tắt Abstractive nguyên bản.")
            return self._run_abstractive_direct(cleaned_text, max_target_tokens, temperature, num_beams, repetition_penalty)

        condensed_text = self.build_condensed_context(
            text,
            extractive_algo=extractive_algo,
            compression_ratio=compression_ratio,
            use_semantic_chunking=use_semantic_chunking,
        )
        condensed_word_count = count_words(condensed_text)
        logger.info(
            f"✅ [Hybrid Summarizer] Nén xong: {input_word_count} từ ➔ {condensed_word_count} từ "
            f"(Giảm {100.0 * (1.0 - condensed_word_count / max(1, input_word_count)):.1f}% số lượng token đầu vào)"
        )
        logger.info(f"⚡ [Hybrid Summarizer] Abstractive Stage: Sinh chữ bằng mô hình: {self.abstractive_model_key}")
        final_summary = self.summarize_from_condensed(
            condensed_text,
            max_target_tokens,
            temperature,
            num_beams,
            repetition_penalty,
        )
        
        elapsed = time.perf_counter() - t_start
        logger.info(
            f"✨ [Hybrid Summarizer] Hoàn tất tóm tắt nhiều tầng trong {elapsed:.3f}s. "
            f"Kích thước bản tóm tắt cuối: {count_words(final_summary)} từ."
        )
        return final_summary

    def _run_abstractive_direct(
        self, 
        text: str, 
        max_length: int, 
        temperature: float, 
        num_beams: int, 
        repetition_penalty: float
    ) -> str:
        """Thực hiện chạy tóm tắt Abstractive trực tiếp từ model engine."""
        try:
            summary = self.abstractive_engine.summarize(
                text=text,
                max_output_length=max_length,
                temperature=temperature,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty
            )
            return summary
        except Exception as exc:
            logger.error(f"Lỗi khi thực thi mô hình Abstractive: {exc}. Fallback sang tóm tắt Extractive...")
            # Fallback sang lấy extractive 3 câu chính của đoạn text
            fallback_runner = EXTRACTIVE_RUNNERS["textrank"]
            return fallback_runner(text, sentence_count=3)


def summarize_retrieved_chunks(
    chunks: list[dict],
    *,
    query: str = "",
    compression_ratio: float = 0.40,
    max_target_tokens: int = 280,
) -> tuple[str, str | None, str | None]:
    """
    Sinh Hybrid Summary từ danh sách RAG chunks đã retrieve + rerank.
    Reuse HybridSummarizer — chọn backbone abstractive tốt nhất đã load.

    Returns:
        (summary_text, model_used, hybrid_algo_key)
    """
    if not chunks:
        return "", None, None

    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        filename = chunk.get("filename", "?")
        page = chunk.get("page")
        page_info = f" trang {page}" if page else ""
        parts.append(f"[Nguồn {i} — {filename}{page_info}]\n{chunk.get('text', '')}")
    combined_text = "\n\n".join(parts)

    try:
        from backend.services.rag.context_compression import pick_best_hybrid_key
        hybrid_key, ext_algo, backbone = pick_best_hybrid_key()
    except Exception:
        hybrid_key, ext_algo, backbone = "textrank-bartpho", "textrank", "bartpho"

    engine = HybridSummarizer(abstractive_model_key=backbone)
    summary = engine.summarize(
        combined_text,
        extractive_algo=ext_algo,
        compression_ratio=compression_ratio,
        max_target_tokens=max_target_tokens,
        temperature=0.15,
        num_beams=4,
        repetition_penalty=1.5,
    )
    model_used = backbone if summary else None
    return summary.strip(), model_used, hybrid_key
