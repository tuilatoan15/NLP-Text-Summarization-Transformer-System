from __future__ import annotations

import re
import numpy as np
from typing import Any


class ChunkingPipeline:
    sentence_pattern = re.compile(r"(?<=[\.\!\?])\s+")

    def split(
        self,
        *,
        text: str,
        pages: list[dict[str, Any]],
        chunk_size: int,
        chunk_overlap: int,
        document_id: str,
        filename: str,
        embedding_service: Any = None,
        embedding_model: str = "intfloat/multilingual-e5-large",
        chunking_mode: str = "dynamic",  # "fixed", "dynamic", "sliding_window"
        threshold: float = 0.6,
        dynamic_k: float = 1.2,
        window_size: int = 2
    ) -> list[dict[str, Any]]:
        # 1. Tách văn bản thành danh sách câu
        raw_sentences = [s.strip() for s in self.sentence_pattern.split(text) if s.strip()]
        if not raw_sentences:
            return []

        # 2. Embedding tất cả các câu
        if embedding_service is None:
            from .embedding_service import EmbeddingService
            embedding_service = EmbeddingService()

        try:
            vectors = embedding_service.embed_documents(raw_sentences, embedding_model)
        except Exception as e:
            # Fallback về split theo từ/paragraph cũ nếu không gọi được model
            return self._fallback_split(
                text=text,
                pages=pages,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                document_id=document_id,
                filename=filename
            )

        X = np.array(vectors, dtype=np.float32)
        n_sentences = len(raw_sentences)

        if n_sentences <= 2:
            chunk_text = " ".join(raw_sentences)
            page = self._resolve_page(chunk_text, pages)
            return [{
                "id": f"{document_id}_chunk_0",
                "document_id": document_id,
                "filename": filename,
                "text": chunk_text,
                "page": page,
                "chunk_index": 0,
                "metadata": {
                    "chunking_mode": "fallback",
                    "sentence_count": n_sentences
                }
            }]

        # 3. Tính toán cosine similarities giữa các câu kề nhau
        similarities = []
        if chunking_mode == "sliding_window":
            for i in range(n_sentences - 1):
                prev_indices = list(range(max(0, i - window_size + 1), i + 1))
                next_indices = list(range(i + 1, min(n_sentences, i + 1 + window_size)))
                
                v_prev = X[prev_indices].mean(axis=0)
                v_next = X[next_indices].mean(axis=0)
                
                norm_prev = np.linalg.norm(v_prev)
                norm_next = np.linalg.norm(v_next)
                if norm_prev == 0 or norm_next == 0:
                    sim = 0.0
                else:
                    sim = float(np.dot(v_prev, v_next) / (norm_prev * norm_next))
                similarities.append(sim)
        else:
            # Chế độ fixed hoặc dynamic thông thường
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            X_norm = X / norms
            similarities = np.sum(X_norm[:-1] * X_norm[1:], axis=1).tolist()

        similarities = np.array(similarities, dtype=np.float32)

        # 4. Xác định các breakpoints ngắt ngữ nghĩa
        breakpoints = set()
        if chunking_mode == "fixed":
            for i, sim in enumerate(similarities):
                if sim < threshold:
                    breakpoints.add(i)
        else:
            # "dynamic" (hoặc mặc định cho sliding_window)
            distances = 1.0 - similarities
            if len(distances) > 0:
                mean_d = np.mean(distances)
                std_d = np.std(distances)
                T = mean_d + dynamic_k * std_d
                for i, dist in enumerate(distances):
                    if dist > T:
                        breakpoints.add(i)

        # 5. Gom các câu lại thành các chunks
        chunks: list[dict[str, Any]] = []
        current_chunk_sentences = []
        current_len = 0
        chunk_index = 0

        for idx, sentence in enumerate(raw_sentences):
            sentence_len = len(sentence)
            is_breakpoint = (idx - 1) in breakpoints if idx > 0 else False

            if current_chunk_sentences and (is_breakpoint or (current_len + sentence_len > chunk_size)):
                chunk_text = " ".join(current_chunk_sentences)
                page = self._resolve_page(chunk_text, pages)
                chunks.append({
                    "id": f"{document_id}_chunk_{chunk_index}",
                    "document_id": document_id,
                    "filename": filename,
                    "text": chunk_text,
                    "page": page,
                    "chunk_index": chunk_index,
                    "metadata": {
                        "chunking_mode": chunking_mode,
                        "sentence_count": len(current_chunk_sentences),
                        "chunk_size": chunk_size
                    }
                })
                chunk_index += 1
                current_chunk_sentences = []
                current_len = 0

            current_chunk_sentences.append(sentence)
            current_len += sentence_len + 1

        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            page = self._resolve_page(chunk_text, pages)
            chunks.append({
                "id": f"{document_id}_chunk_{chunk_index}",
                "document_id": document_id,
                "filename": filename,
                "text": chunk_text,
                "page": page,
                "chunk_index": chunk_index,
                "metadata": {
                    "chunking_mode": chunking_mode,
                    "sentence_count": len(current_chunk_sentences),
                    "chunk_size": chunk_size
                }
            })

        return chunks

    def _fallback_split(
        self,
        *,
        text: str,
        pages: list[dict[str, Any]],
        chunk_size: int,
        chunk_overlap: int,
        document_id: str,
        filename: str,
    ) -> list[dict[str, Any]]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[dict[str, Any]] = []
        cursor = 0
        chunk_index = 0

        while cursor < len(paragraphs):
            candidate = []
            current_len = 0
            look_ahead = cursor
            while look_ahead < len(paragraphs):
                paragraph = paragraphs[look_ahead]
                if current_len + len(paragraph) > chunk_size and candidate:
                    break
                candidate.append(paragraph)
                current_len += len(paragraph) + 2
                look_ahead += 1

            raw_chunk = "\n\n".join(candidate)
            chunk_text = self._semantic_compact(raw_chunk, chunk_size)
            page = self._resolve_page(chunk_text, pages)
            chunks.append({
                "id": f"{document_id}_chunk_{chunk_index}",
                "document_id": document_id,
                "filename": filename,
                "text": chunk_text,
                "page": page,
                "chunk_index": chunk_index,
                "metadata": {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "fallback": True,
                },
            })

            chunk_index += 1
            if look_ahead >= len(paragraphs):
                break
            if chunk_overlap <= 0:
                cursor = look_ahead
            else:
                overlap_chars = 0
                overlap_steps = 0
                back = look_ahead - 1
                while back >= cursor and overlap_chars < chunk_overlap:
                    overlap_chars += len(paragraphs[back]) + 2
                    overlap_steps += 1
                    back -= 1
                cursor = max(look_ahead - overlap_steps, cursor + 1)
        return chunks

    def _semantic_compact(self, text: str, chunk_size: int) -> str:
        if len(text) <= chunk_size:
            return text
        sentences = [s.strip() for s in self.sentence_pattern.split(text) if s.strip()]
        output: list[str] = []
        total = 0
        for sentence in sentences:
            if total + len(sentence) > chunk_size and output:
                break
            output.append(sentence)
            total += len(sentence) + 1
        return " ".join(output) if output else text[:chunk_size]

    def _resolve_page(self, chunk_text: str, pages: list[dict[str, Any]]) -> int | None:
        if not pages:
            return None
        for page in pages:
            page_text = page.get("text", "")
            if chunk_text[:80] and chunk_text[:80] in page_text:
                return int(page.get("page", 1))
        return int(pages[0].get("page", 1))

