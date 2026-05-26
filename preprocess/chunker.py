"""Semantic, token-aware chunking optimized for factual summarization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from pipeline.schema import ChunkingConfig, DocumentElement, TextChunk
from preprocess.tokenizer import VietnameseTokenizer
from utils.logger import logger


@dataclass(slots=True)
class ChunkUnit:
    text: str
    element_index: int
    element_type: str
    token_count: int
    page_number: int | None
    section_path: list[str]


class SemanticChunker:
    """Build coherent chunks using heading, paragraph, sentence, and semantic boundaries."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        self.tokenizer = VietnameseTokenizer(
            token_model_name=self.config.token_model_name,
            use_vietnamese_segmenter=self.config.use_vietnamese_segmenter,
        )

    def chunk(self, document_id: str, elements: list[DocumentElement], fallback_text: str = "") -> list[TextChunk]:
        units = self._build_units(elements, fallback_text)
        if not units:
            return []

        boundary_scores = self._semantic_boundary_scores(units)
        chunks: list[TextChunk] = []
        current: list[ChunkUnit] = []
        current_tokens = 0

        for idx, unit in enumerate(units):
            force_heading_boundary = (
                self.config.respect_headings
                and unit.element_type == "heading"
                and current_tokens >= self.config.min_tokens
            )
            semantic_boundary = (
                idx > 0
                and boundary_scores
                and boundary_scores[idx - 1] < self.config.semantic_similarity_threshold
                and current_tokens >= self.config.min_tokens
            )
            would_exceed = current_tokens + unit.token_count > self._dynamic_max_tokens(current)
            if current and (force_heading_boundary or semantic_boundary or would_exceed):
                chunks.append(self._make_chunk(document_id, chunks, current))
                current = self._overlap_units(current)
                current_tokens = sum(item.token_count for item in current)

            current.append(unit)
            current_tokens += unit.token_count

        if current:
            chunks.append(self._make_chunk(document_id, chunks, current))

        return self._dedupe_chunks(chunks)

    def _build_units(self, elements: list[DocumentElement], fallback_text: str) -> list[ChunkUnit]:
        if not elements and fallback_text:
            elements = [DocumentElement(text=fallback_text, element_type="paragraph")]

        units: list[ChunkUnit] = []
        current_section_path: list[str] = []
        for element_index, element in enumerate(elements):
            text = element.text.strip()
            if not text:
                continue
            if element.element_type == "heading":
                level = element.level or max(1, len(element.section_path) or 1)
                current_section_path = (element.section_path or current_section_path[: max(0, level - 1)] + [text]).copy()
            section_path = element.section_path.copy() if element.section_path else current_section_path.copy()
            token_count = self.tokenizer.count_tokens(text)
            if (
                self.config.split_long_paragraphs
                and element.element_type == "paragraph"
                and token_count > self.config.max_tokens
            ):
                units.extend(self._split_long_paragraph(element, element_index, section_path))
            else:
                units.append(
                    ChunkUnit(
                        text=text,
                        element_index=element_index,
                        element_type=element.element_type,
                        token_count=token_count,
                        page_number=element.page_number,
                        section_path=section_path,
                    )
                )
        return units

    def _split_long_paragraph(
        self,
        element: DocumentElement,
        element_index: int,
        section_path: list[str],
    ) -> list[ChunkUnit]:
        sentences = self.tokenizer.split_sentences(element.text)
        units: list[ChunkUnit] = []
        buffer: list[str] = []
        buffer_tokens = 0
        for sentence in sentences:
            sentence_tokens = self.tokenizer.count_tokens(sentence)
            if buffer and buffer_tokens + sentence_tokens > self.config.target_tokens:
                text = " ".join(buffer).strip()
                units.append(
                    ChunkUnit(
                        text=text,
                        element_index=element_index,
                        element_type="paragraph",
                        token_count=self.tokenizer.count_tokens(text),
                        page_number=element.page_number,
                        section_path=section_path.copy(),
                    )
                )
                buffer = []
                buffer_tokens = 0
            buffer.append(sentence)
            buffer_tokens += sentence_tokens
        if buffer:
            text = " ".join(buffer).strip()
            units.append(
                ChunkUnit(
                    text=text,
                    element_index=element_index,
                    element_type="paragraph",
                    token_count=self.tokenizer.count_tokens(text),
                    page_number=element.page_number,
                    section_path=section_path.copy(),
                )
            )
        return units

    def _semantic_boundary_scores(self, units: list[ChunkUnit]) -> list[float]:
        if not self.config.semantic_model_name or len(units) < 2:
            return []
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.config.semantic_model_name)
            embeddings = model.encode(
                [unit.text for unit in units],
                batch_size=16,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vectors = np.asarray(embeddings, dtype=np.float32)
            scores: list[float] = []
            for idx in range(len(units) - 1):
                left = vectors[idx]
                right = vectors[idx + 1]
                denom = np.linalg.norm(left) * np.linalg.norm(right)
                scores.append(float(np.dot(left, right) / denom) if denom else 0.0)
            return scores
        except Exception as exc:
            logger.warning("Semantic boundary model unavailable, falling back to structural chunking: %s", exc)
            return []

    def _dynamic_max_tokens(self, current: list[ChunkUnit]) -> int:
        if not self.config.dynamic_size or not current:
            return self.config.max_tokens
        has_table = any(unit.element_type == "table" for unit in current)
        has_heading = any(unit.element_type == "heading" for unit in current)
        if has_table:
            return min(self.config.max_tokens, self.config.target_tokens + 120)
        if has_heading:
            return self.config.max_tokens
        return max(self.config.min_tokens, self.config.target_tokens)

    def _overlap_units(self, units: list[ChunkUnit]) -> list[ChunkUnit]:
        if self.config.overlap_tokens <= 0:
            return []
        overlap: list[ChunkUnit] = []
        total = 0
        for unit in reversed(units):
            if unit.element_type == "heading":
                if overlap:
                    overlap.insert(0, unit)
                continue
            remaining = self.config.overlap_tokens - total
            if remaining <= 0:
                break
            if unit.token_count <= remaining:
                overlap.insert(0, unit)
                total += unit.token_count
            else:
                tail = self._tail_overlap_unit(unit, remaining)
                if tail is not None:
                    overlap.insert(0, tail)
                    total += tail.token_count
                break
            if total >= self.config.overlap_tokens:
                break
        return overlap

    def _tail_overlap_unit(self, unit: ChunkUnit, max_tokens: int) -> ChunkUnit | None:
        if max_tokens <= 0:
            return None
        sentences = self.tokenizer.split_sentences(unit.text)
        selected: list[str] = []
        total = 0
        for sentence in reversed(sentences):
            count = self.tokenizer.count_tokens(sentence)
            if selected and total + count > max_tokens:
                break
            if count > max_tokens * 2 and not selected:
                return None
            selected.insert(0, sentence)
            total += count
            if total >= max_tokens:
                break
        if not selected:
            return None
        text = " ".join(selected).strip()
        return ChunkUnit(
            text=text,
            element_index=unit.element_index,
            element_type=unit.element_type,
            token_count=self.tokenizer.count_tokens(text),
            page_number=unit.page_number,
            section_path=unit.section_path.copy(),
        )

    def _make_chunk(self, document_id: str, chunks: list[TextChunk], units: list[ChunkUnit]) -> TextChunk:
        text = self._join_units(units)
        token_count = self.tokenizer.count_tokens(text)
        pages = [unit.page_number for unit in units if unit.page_number]
        section_path = self._dominant_section_path(units)
        index = len(chunks)
        chunk_id = self._chunk_id(document_id, index, text)
        return TextChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text=text,
            index=index,
            token_count=token_count,
            word_count=len(text.split()),
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            section_path=section_path,
            source_element_ids=list(dict.fromkeys(unit.element_index for unit in units)),
            overlap_from_previous=index > 0 and bool(units),
            metadata={
                "element_types": list(dict.fromkeys(unit.element_type for unit in units)),
                "sentence_count": len(self.tokenizer.split_sentences(text)),
            },
        )

    @staticmethod
    def _join_units(units: Iterable[ChunkUnit]) -> str:
        parts: list[str] = []
        for unit in units:
            if unit.element_type == "heading":
                parts.append(unit.text.strip())
            elif unit.element_type == "bullet":
                text = unit.text.strip()
                parts.append(text if text.startswith(("-", "*", "•")) else f"- {text}")
            else:
                parts.append(unit.text.strip())
        return "\n\n".join(part for part in parts if part).strip()

    @staticmethod
    def _dominant_section_path(units: list[ChunkUnit]) -> list[str]:
        for unit in reversed(units):
            if unit.section_path:
                return unit.section_path.copy()
        return []

    @staticmethod
    def _chunk_id(document_id: str, index: int, text: str) -> str:
        digest = hashlib.sha1(f"{document_id}:{index}:{text[:512]}".encode("utf-8", errors="ignore")).hexdigest()
        return digest[:20]

    @staticmethod
    def _dedupe_chunks(chunks: list[TextChunk]) -> list[TextChunk]:
        deduped: list[TextChunk] = []
        seen: set[str] = set()
        for chunk in chunks:
            key = " ".join(chunk.text.lower().split())
            if key in seen:
                continue
            seen.add(key)
            chunk.index = len(deduped)
            deduped.append(chunk)
        return deduped
