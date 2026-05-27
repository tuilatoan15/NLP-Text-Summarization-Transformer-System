from __future__ import annotations

import re
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
            chunks.append(
                {
                    "id": f"{document_id}_chunk_{chunk_index}",
                    "document_id": document_id,
                    "filename": filename,
                    "text": chunk_text,
                    "page": page,
                    "chunk_index": chunk_index,
                    "metadata": {
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "semantic_compacted": True,
                    },
                }
            )

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

