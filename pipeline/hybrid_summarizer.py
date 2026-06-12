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
        
        # Tiền xử lý văn bản đầu vào
        cleaned_text = clean_text(text, aggressive=True)
        sentences = split_sentences(cleaned_text)
        
        input_word_count = count_words(cleaned_text)
        if input_word_count < 10 or len(sentences) <= 3:
            logger.info("Văn bản quá ngắn, chuyển trực tiếp sang tóm tắt Abstractive nguyên bản.")
            return self._run_abstractive_direct(cleaned_text, max_target_tokens, temperature, num_beams, repetition_penalty)

        # Lấy số câu cần chọn lọc
        num_sentences = max(3, min(int(len(sentences) * compression_ratio), 25))

        # ── Bước 1: Filtering (Nén văn bản) ───────────────────
        if use_semantic_chunking:
            logger.info(f"⚡ [Hybrid Summarizer] Semantic Chunking Stage: Lọc lấy top {num_sentences} câu")
            chunker = SemanticChunker(threshold=0.45)
            semantic_chunks = chunker.chunk_document(sentences)
            
            selected_sentences = []
            sents_per_chunk = max(1, int(num_sentences / max(1, len(semantic_chunks))))
            
            for chunk in semantic_chunks:
                chunk_sents = split_sentences(chunk)
                if len(chunk_sents) <= sents_per_chunk:
                    selected_sentences.extend(chunk_sents)
                else:
                    extractive_runner = EXTRACTIVE_RUNNERS.get(extractive_algo) or EXTRACTIVE_RUNNERS["textrank"]
                    details = extractive_runner(chunk, sentence_count=sents_per_chunk)
                    selected_sentences.extend(split_sentences(details.get("summary", "")))
            
            condensed_text = " ".join(selected_sentences[:num_sentences])
        else:
            logger.info(
                f"⚡ [Hybrid Summarizer] Extractive Stage: {len(sentences)} câu ➔ Lọc lấy top {num_sentences} câu "
                f"bằng thuật toán '{extractive_algo}' (tỷ lệ nén {compression_ratio * 100:.1f}%)"
            )
            
            extractive_runner = EXTRACTIVE_RUNNERS.get(extractive_algo)
            if not extractive_runner:
                logger.warning(f"Thuật toán extractive '{extractive_algo}' không tìm thấy. Fallback sang 'textrank'")
                extractive_runner = EXTRACTIVE_RUNNERS["textrank"]
                
            condensed_details = extractive_runner(cleaned_text, sentence_count=num_sentences)
            condensed_text = condensed_details.get("summary", "")
        
        condensed_word_count = count_words(condensed_text)
        logger.info(
            f"✅ [Hybrid Summarizer] Nén xong: {input_word_count} từ ➔ {condensed_word_count} từ "
            f"(Giảm {100.0 * (1.0 - condensed_word_count / input_word_count):.1f}% số lượng token đầu vào)"
        )

        # ── Bước 2: Abstractive Generation (Sinh văn bản) ──────────────────
        logger.info(f"⚡ [Hybrid Summarizer] Abstractive Stage: Sinh chữ bằng mô hình: {self.abstractive_model_key}")
        
        final_summary = self._run_abstractive_direct(
            condensed_text, 
            max_target_tokens, 
            temperature, 
            num_beams, 
            repetition_penalty
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
