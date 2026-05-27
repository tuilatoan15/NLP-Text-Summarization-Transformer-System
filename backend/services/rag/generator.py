from __future__ import annotations

import re


class GroundedGenerator:
    insufficient_context_message = "Không tìm thấy thông tin trong tài liệu"

    def build_answer(self, query: str, contexts: list[dict], temperature: float = 0.2) -> dict:
        if not contexts:
            return {
                "answer": self.insufficient_context_message,
                "confidence": 0.0,
                "grounded": True,
            }

        best = contexts[0]
        confidence = min(0.99, max(0.0, best["combined_score"]))
        answer = self._extract_relevant_sentences(query, [c["text"] for c in contexts], max_sentences=4)
        if not answer.strip():
            answer = self.insufficient_context_message
            confidence = 0.0
        return {
            "answer": answer,
            "confidence": round(confidence, 4),
            "grounded": True,
            "temperature_used": temperature,
        }

    def _extract_relevant_sentences(self, query: str, passages: list[str], max_sentences: int) -> str:
        query_terms = {t.lower() for t in re.findall(r"\w+", query)}
        sentences: list[tuple[float, str]] = []
        for passage in passages:
            for sentence in re.split(r"(?<=[\.\!\?])\s+", passage):
                s = sentence.strip()
                if not s:
                    continue
                terms = {t.lower() for t in re.findall(r"\w+", s)}
                overlap = len(query_terms.intersection(terms))
                if overlap > 0:
                    sentences.append((float(overlap), s))
        sentences.sort(key=lambda item: item[0], reverse=True)
        selected = [s for _, s in sentences[:max_sentences]]
        return "\n".join(selected)

    def prompt_template(self, contexts: list[dict], question: str) -> str:
        blocks = []
        for idx, c in enumerate(contexts, start=1):
            blocks.append(
                f"[Nguon {idx}] file={c['filename']} page={c.get('page')} score={c['combined_score']}\n{c['text']}"
            )
        joined = "\n\n".join(blocks)
        return (
            "Ban la tro ly RAG. Chi tra loi dua tren CONTEXT. "
            "Neu khong du thong tin, phai tra loi: 'Không tìm thấy thông tin trong tài liệu'.\n\n"
            f"CONTEXT:\n{joined}\n\nQUESTION: {question}\nANSWER:"
        )

