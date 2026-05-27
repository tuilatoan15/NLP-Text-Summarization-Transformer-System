from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np


token_pattern = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in token_pattern.findall(text or "")]


class HybridRetriever:
    def retrieve(
        self,
        *,
        query: str,
        query_vector: list[float],
        chunks: list[dict[str, Any]],
        top_k: int,
        threshold: float,
        retrieval_mode: str = "hybrid",
        use_reranking: bool = False,
    ) -> list[dict[str, Any]]:
        if not chunks:
            return []
        bm25 = self._bm25_scores(query, chunks)
        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q) or 1.0

        scored: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            if "embedding_score" in chunk:
                sim = float(chunk["embedding_score"])
            else:
                vec = np.array(chunk["vector"], dtype=np.float32)
                sim = float(np.dot(q, vec) / ((np.linalg.norm(vec) or 1.0) * q_norm))
            bm = bm25[idx]
            if retrieval_mode == "embedding":
                combined = sim
            elif retrieval_mode == "bm25":
                combined = bm
            else:
                combined = 0.65 * sim + 0.35 * bm
            if combined >= threshold:
                scored.append(
                    {
                        "chunk_id": chunk["id"],
                        "document_id": chunk["document_id"],
                        "filename": chunk["filename"],
                        "page": chunk.get("page"),
                        "text": chunk["text"],
                        "embedding_score": round(sim, 6),
                        "bm25_score": round(bm, 6),
                        "combined_score": round(combined, 6),
                    }
                )

        scored.sort(key=lambda item: item["combined_score"], reverse=True)
        top = scored[:top_k]

        if use_reranking and top:
            top.sort(
                key=lambda item: (self._term_coverage(query, item["text"]), item["combined_score"]),
                reverse=True,
            )

        for rank, item in enumerate(top, start=1):
            item["rank"] = rank
        return top

    def _term_coverage(self, query: str, text: str) -> float:
        q_terms = set(_tokens(query))
        if not q_terms:
            return 0.0
        t_terms = set(_tokens(text))
        return len(q_terms.intersection(t_terms)) / len(q_terms)

    def _bm25_scores(self, query: str, chunks: list[dict[str, Any]]) -> list[float]:
        docs_tokens = [_tokens(chunk["text"]) for chunk in chunks]
        query_tokens = _tokens(query)
        if not query_tokens:
            return [0.0] * len(chunks)

        avgdl = sum(len(tokens) for tokens in docs_tokens) / max(len(docs_tokens), 1)
        df: dict[str, int] = defaultdict(int)
        for tokens in docs_tokens:
            for term in set(tokens):
                df[term] += 1
        n_docs = len(docs_tokens)
        k1 = 1.5
        b = 0.75

        scores: list[float] = []
        for tokens in docs_tokens:
            tf = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for term in query_tokens:
                if term not in tf:
                    continue
                idf = math.log((n_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1.0)
                numer = tf[term] * (k1 + 1)
                denom = tf[term] + k1 * (1 - b + b * (dl / (avgdl or 1.0)))
                score += idf * (numer / (denom or 1.0))
            scores.append(float(score))

        max_score = max(scores) if scores else 0.0
        if max_score > 0:
            scores = [s / max_score for s in scores]
        return scores

