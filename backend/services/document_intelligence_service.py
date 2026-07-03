"""Document Intelligence service for research-grade summarization workflows."""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from embeddings.embedder import SentenceTransformerEmbedder
from embeddings.vector_store import VectorIndex, get_vector_store
from pipeline.ingest_pipeline import IngestPipeline
from visualization.embeddings_viz import embedding_map_2d, similarity_heatmap
from pipeline.schema import EmbeddingConfig, IngestConfig
from src.extractive import summarize_extractive_algorithm
from src.preprocess import clean_text, split_sentences, tokenize_words
from src.utils import count_words, logger, resolve_torch_device_str, save_json
from utils.metrics import lexical_overlap


STORE_DIR = Path("storage/document_intelligence")
DEFAULT_ANALYSIS_ALGORITHMS = ["textrank", "lexrank", "lsa", "tfidf", "vit5", "bartpho"]
VI_STOPWORDS = {
    "và", "của", "các", "những", "một", "trong", "cho", "với", "được", "đã",
    "là", "có", "không", "này", "đó", "từ", "khi", "về", "theo", "sau",
    "trước", "tại", "để", "nhiều", "người", "năm", "ngày", "ra", "vào",
    "trên", "dưới", "đến", "bằng", "hoặc", "như", "cũng", "nên", "phải",
}


@dataclass(slots=True)
class DocumentRecord:
    document_id: str
    payload: dict[str, Any]


class DocumentIntelligenceService:
    """High-level orchestration layer for AI document analysis."""

    def __init__(self, store_dir: str | Path = STORE_DIR) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._cache = None

    def ingest_file(
        self,
        path: str | Path,
        *,
        include_embeddings: bool = True,
        embedding_model: str | None = None,
    ) -> dict[str, Any]:
        config = self._ingest_config(embedding_model)
        config.enable_embeddings = include_embeddings
        start = time.perf_counter()
        result = IngestPipeline(config).ingest(path, include_embeddings=include_embeddings)
        payload = result.to_dict()
        payload["analysis_assets"] = self.generate_assets(payload)
        payload["visualization"] = self.build_visualization(payload)
        payload["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        payload["ingest_seconds"] = round(time.perf_counter() - start, 4)
        self._persist_vectors(payload)
        self._save(payload)
        return payload

    def get_document(self, document_id: str) -> dict[str, Any]:
        cache_key = f"docintel:doc:{document_id}"
        cached = self._cache_service().get_json(cache_key)
        if cached:
            return cached
        path = self._record_path(document_id)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {document_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._cache_service().set_json(cache_key, payload, ttl_seconds=600)
        return payload

    def _cache_service(self):
        if self._cache is None:
            from backend.services.cache_service import CacheService

            self._cache = CacheService()
        return self._cache

    def list_documents(self, limit: int = 50) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in sorted(self.store_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                {
                    "document_id": payload.get("document_id"),
                    "title": payload.get("metadata", {}).get("title"),
                    "source_type": payload.get("metadata", {}).get("source_type"),
                    "created_at": payload.get("created_at"),
                    "chunk_count": len(payload.get("chunks") or []),
                    "word_count": payload.get("quality", {}).get("extraction", {}).get("word_count"),
                    "quality_score": payload.get("quality", {}).get("extraction", {}).get("score"),
                }
            )
        return records

    def semantic_search(self, document_id: str, query: str, top_k: int = 5) -> dict[str, Any]:
        payload = self.get_document(document_id)
        chunks = payload.get("chunks") or []
        if not chunks:
            return {"query": query, "results": []}

        embeddings = payload.get("embeddings")
        results: list[dict[str, Any]]
        retrieval_backend = "lexical"
        if embeddings:
            vectors = np.asarray(embeddings, dtype=np.float32)
            query_vector = self._embed_query(payload, query, vectors.shape[1])
            if query_vector is not None and query_vector.shape[0] == vectors.shape[1]:
                index = VectorIndex(vectors)
                hits = index.search(query_vector, top_k=top_k)
                results = [
                    {
                        "rank": rank,
                        "score": round(hit.score, 4),
                        "chunk": chunks[hit.index],
                        "highlight": self._query_highlight(query, chunks[hit.index].get("text", "")),
                    }
                    for rank, hit in enumerate(hits, start=1)
                    if hit.index < len(chunks)
                ]
                retrieval_backend = index.backend
            else:
                results = self._lexical_chunk_results(chunks, query, top_k)
        else:
            results = self._lexical_chunk_results(chunks, query, top_k)

        return {
            "query": query,
            "top_k": top_k,
            "results": results,
            "retrieval_mode": "embedding" if embeddings else "lexical",
            "retrieval_backend": retrieval_backend,
        }

    def compare_summaries(
        self,
        document_id: str,
        *,
        reference: str | None = None,
        algorithms: list[str] | None = None,
        target_length_ratio: int = 20,
        extractive_sentences: int = 5,
        max_abstractive_length: int = 180,
    ) -> dict[str, Any]:
        payload = self.get_document(document_id)
        source_text = payload.get("clean_text") or ""
        selected = algorithms or DEFAULT_ANALYSIS_ALGORITHMS
        from src.dashboard import summarize_all

        compare = summarize_all(
            text=source_text,
            reference=reference,
            algorithms=selected,
            sentence_count=extractive_sentences,
            max_output_length=max_abstractive_length,
            target_length_ratio=target_length_ratio,
            use_length_ratio=True,
        )

        chunks = payload.get("chunks") or []
        from evaluation.hallucination import audit_summary

        for row in compare.get("results", []):
            row["citations"] = self.ground_summary(row.get("summary", ""), chunks)
            row["consistency"] = audit_summary(
                row.get("summary", ""),
                source_text,
                chunks=chunks,
                mode="fast",
            )

        compare["document_id"] = document_id
        compare["document_metadata"] = payload.get("metadata", {})
        compare["research_matrix"] = self._research_matrix(compare.get("results", []))
        compare["summary_tree"] = self.build_summary_tree(payload, compare)
        compare["visualization"] = {
            **payload.get("visualization", {}),
            "comparison_radar": compare.get("charts", {}).get("radar", []),
            "latency": compare.get("charts", {}).get("time", []),
        }
        compare["system_metrics"] = self._system_metrics_snapshot()
        self._save_analysis(document_id, compare)
        return compare

    def hierarchical_summarize(
        self,
        document_id: str,
        *,
        model_key: str = "vit5",
        use_extractive_map: bool = False,
    ) -> dict[str, Any]:
        payload = self.get_document(document_id)
        from summarizers.hierarchical import hierarchical_summarize

        result = hierarchical_summarize(
            payload.get("clean_text") or "",
            payload.get("chunks") or [],
            model_key=model_key,
            use_extractive_map=use_extractive_map,
        )
        result["document_id"] = document_id
        return result

    def explain_extractive(self, document_id: str, algorithm: str = "textrank") -> dict[str, Any]:
        payload = self.get_document(document_id)
        source = payload.get("clean_text") or ""
        from src.explainability import build_extractive_explanations, build_sentence_ranking_graph

        graph = build_sentence_ranking_graph(source, algorithm=algorithm)
        explanations = build_extractive_explanations(source, graph.get("summary", ""))
        return {
            "document_id": document_id,
            "algorithm": algorithm,
            "ranking_graph": graph,
            "explanations": explanations,
        }

    @staticmethod
    def _system_metrics_snapshot() -> dict[str, Any]:
        import os

        snapshot: dict[str, Any] = {"cpu_count": os.cpu_count()}
        try:
            import psutil

            proc = psutil.Process()
            snapshot["memory_usage_mb"] = round(proc.memory_info().rss / (1024 * 1024), 2)
            snapshot["cpu_percent"] = proc.cpu_percent(interval=0.05)
        except Exception:
            snapshot["memory_usage_mb"] = None
        try:
            import torch

            if torch.cuda.is_available():
                snapshot["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        except Exception:
            pass
        return snapshot

    def generate_assets(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = payload.get("clean_text") or ""
        chunks = payload.get("chunks") or []
        sentences = split_sentences(text)
        overview = summarize_extractive_algorithm(text, "textrank", sentence_count=min(5, max(1, len(sentences))))["summary"]
        keywords = self._keywords(text, limit=18)
        key_takeaways = self._takeaways(text, limit=6)
        entities = self._entities(text)
        timeline = self._timeline(sentences)

        return {
            "overview": {
                "document_overview": overview,
                "key_insights": key_takeaways[:4],
                "keywords": keywords[:12],
                "source_words": count_words(text),
                "chunk_count": len(chunks),
            },
            "reports": self._research_report(payload, overview, key_takeaways),
            "quiz": self._quiz(key_takeaways, keywords),
            "flashcards": self._flashcards(key_takeaways, keywords),
            "mindmap": self._mindmap(payload, keywords),
            "presentation": self._presentation(payload, key_takeaways),
            "podcast": self._podcast_script(payload, key_takeaways),
            "infographic": self._infographic(payload, keywords, entities),
            "timeline": timeline,
            "entity_graph": self._entity_graph(entities, keywords),
        }

    def build_visualization(self, payload: dict[str, Any]) -> dict[str, Any]:
        chunks = payload.get("chunks") or []
        embeddings = payload.get("embeddings")
        return {
            "chunk_graph": self._chunk_graph(chunks),
            "embedding_map": embedding_map_2d(chunks, embeddings),
            "similarity_heatmap": similarity_heatmap(embeddings),
            "chunk_hierarchy": self._chunk_hierarchy(chunks),
        }

    def build_summary_tree(self, payload: dict[str, Any], compare: dict[str, Any]) -> dict[str, Any]:
        chunks = payload.get("chunks") or []
        best_key = (compare.get("best_model") or {}).get("key")
        best_row = next((row for row in compare.get("results", []) if row.get("key") == best_key), None)
        chunk_summaries = []
        for chunk in chunks[:24]:
            text = chunk.get("text", "")
            summary = summarize_extractive_algorithm(text, "textrank", sentence_count=1)["summary"] if text else ""
            chunk_summaries.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "section_path": chunk.get("section_path", []),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "summary": summary,
                }
            )
        section_map: dict[str, list[str]] = {}
        for item in chunk_summaries:
            key = " / ".join(item.get("section_path") or ["Unsectioned"])
            section_map.setdefault(key, []).append(item.get("summary", ""))
        return {
            "chunk_summaries": chunk_summaries,
            "section_summaries": [
                {"section": section, "summary": clean_text(" ".join(parts))}
                for section, parts in section_map.items()
            ],
            "global_summary": (best_row or {}).get("summary", ""),
            "best_model": best_key,
        }

    def ground_summary(self, summary: str, chunks: list[dict[str, Any]], top_k: int = 2) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        for sentence_index, sentence in enumerate(split_sentences(summary)):
            scored = []
            for chunk in chunks:
                score = lexical_overlap(sentence, chunk.get("text", ""))
                scored.append((score, chunk))
            scored.sort(key=lambda item: item[0], reverse=True)
            evidence = []
            for score, chunk in scored[:top_k]:
                evidence.append(
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "page_start": chunk.get("page_start"),
                        "page_end": chunk.get("page_end"),
                        "section_path": chunk.get("section_path", []),
                        "support_score": round(float(score), 4),
                        "excerpt": self._best_excerpt(sentence, chunk.get("text", "")),
                    }
                )
            best_score = evidence[0]["support_score"] if evidence else 0.0
            citations.append(
                {
                    "sentence_index": sentence_index,
                    "sentence": sentence,
                    "status": "grounded" if best_score >= 0.35 else "needs_review",
                    "best_support_score": best_score,
                    "evidence": evidence,
                }
            )
        return citations

    def _ingest_config(self, embedding_model: str | None) -> IngestConfig:
        path = Path("configs/ingest.json")
        config = IngestConfig.from_json(path) if path.exists() else IngestConfig()
        config.embedding.show_progress = False
        config.embedding.fallback_to_hashing = True
        if embedding_model:
            config.embedding.model_name = embedding_model
        return config

    def _persist_vectors(self, payload: dict[str, Any]) -> None:
        embeddings = payload.get("embeddings")
        chunks = payload.get("chunks") or []
        if not embeddings or not chunks:
            return
        try:
            vectors = np.asarray(embeddings, dtype=np.float32)
            store = get_vector_store()
            chunk_ids = [c.get("chunk_id", str(i)) for i, c in enumerate(chunks)]
            metadatas = [
                {
                    "page_start": c.get("page_start"),
                    "section_path": c.get("section_path", []),
                }
                for c in chunks
            ]
            store.upsert(payload["document_id"], chunk_ids, vectors, metadatas)
            payload["vector_backend"] = type(store).__name__
        except Exception as exc:
            logger.warning("Vector store persistence skipped: %s", exc)

    def _save(self, payload: dict[str, Any]) -> None:
        save_json(payload, self._record_path(payload["document_id"]))

    def _save_analysis(self, document_id: str, payload: dict[str, Any]) -> None:
        target = self.store_dir / f"{document_id}.analysis.json"
        save_json(payload, target)

    def _record_path(self, document_id: str) -> Path:
        return self.store_dir / f"{document_id}.json"

    def _embed_query(self, payload: dict[str, Any], query: str, dimension: int) -> np.ndarray | None:
        model_name = payload.get("quality", {}).get("embedding_model") or payload.get("quality", {}).get("embedding", {}).get("model")
        if model_name in {None, "hash-fallback"} or dimension == 384:
            model_name = "hash"
        try:
            embedder = SentenceTransformerEmbedder(
                EmbeddingConfig(
                    model_name=str(model_name),
                    device=resolve_torch_device_str(),
                    use_fp16=True,
                    show_progress=False,
                    fallback_to_hashing=True,
                    normalize_embeddings=True,
                )
            )
            return embedder.embed_query(query)
        except Exception as exc:
            logger.warning("Query embedding failed, using lexical retrieval: %s", exc)
            return None

    @staticmethod
    def _cosine_scores(query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        query_norm = np.linalg.norm(query_vector) or 1.0
        vector_norms = np.linalg.norm(vectors, axis=1)
        vector_norms[vector_norms == 0] = 1.0
        return (vectors @ query_vector) / (vector_norms * query_norm)

    @staticmethod
    def _ranked_chunk_results(chunks: list[dict[str, Any]], scores: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        order = list(np.argsort(-scores)[:top_k])
        return [
            {
                "rank": rank,
                "score": round(float(scores[idx]), 4),
                "chunk": chunks[idx],
                "highlight": DocumentIntelligenceService._query_highlight("", chunks[idx].get("text", "")),
            }
            for rank, idx in enumerate(order, start=1)
        ]

    @staticmethod
    def _lexical_chunk_results(chunks: list[dict[str, Any]], query: str, top_k: int) -> list[dict[str, Any]]:
        scored = [(lexical_overlap(query, chunk.get("text", "")), chunk) for chunk in chunks]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "rank": rank,
                "score": round(float(score), 4),
                "chunk": chunk,
                "highlight": DocumentIntelligenceService._query_highlight(query, chunk.get("text", "")),
            }
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]

    @staticmethod
    def _query_highlight(query: str, text: str, max_chars: int = 420) -> str:
        del query
        return (text or "")[:max_chars].strip()

    @staticmethod
    def _best_excerpt(sentence: str, text: str) -> str:
        source_sentences = split_sentences(text)
        if not source_sentences:
            return text[:360]
        best = max(source_sentences, key=lambda item: lexical_overlap(sentence, item))
        return best[:420]

    @staticmethod
    def _keywords(text: str, limit: int = 20) -> list[dict[str, Any]]:
        tokens = [
            token.lower()
            for token in tokenize_words(clean_text(text), remove_stopwords=False)
            if len(token) > 2 and token.lower() not in VI_STOPWORDS and not token.isdigit()
        ]
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        return [
            {"term": term, "count": count, "weight": round(count / total, 4)}
            for term, count in counts.most_common(limit)
        ]

    @staticmethod
    def _takeaways(text: str, limit: int = 6) -> list[str]:
        details = summarize_extractive_algorithm(text, "textrank", sentence_count=limit)
        return [item["sentence"] for item in details.get("selected_sentences", [])][:limit]

    @staticmethod
    def _entities(text: str) -> list[dict[str, Any]]:
        patterns = {
            "date": r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|tháng\s+\d{1,2})\b",
            "number": r"\b\d+(?:[,.]\d+)?\s*(?:%|MW|kWh|tỷ|triệu|nghìn|km|ha)?\b",
            "organization": r"\b(?:Tập đoàn|Bộ|Sở|Ủy ban|Công ty|Trường|Viện|EVN|UBND)[^.,;\n]{0,80}",
            "person_or_concept": r"\b[A-ZÀ-ỸĐ][\wÀ-ỹĐđ]+(?:\s+[A-ZÀ-ỸĐ][\wÀ-ỹĐđ]+){1,5}\b",
        }
        entities: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for label, pattern in patterns.items():
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = match.group(0).strip()
                key = (label, value.lower())
                if len(value) < 3 or key in seen:
                    continue
                seen.add(key)
                entities.append({"type": label, "text": value, "start": match.start(), "end": match.end()})
        return entities[:80]

    @staticmethod
    def _timeline(sentences: list[str]) -> list[dict[str, Any]]:
        items = []
        date_re = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|tháng\s+\d{1,2})\b", re.IGNORECASE)
        for sentence in sentences:
            match = date_re.search(sentence)
            if match:
                items.append({"time": match.group(1), "event": sentence})
        return items[:30]

    @staticmethod
    def _research_report(payload: dict[str, Any], overview: str, takeaways: list[str]) -> dict[str, Any]:
        metadata = payload.get("metadata", {})
        return {
            "title": f"Báo cáo phân tích: {metadata.get('title') or payload.get('document_id')}",
            "abstract": overview,
            "methodology": [
                "Structured parsing with PDF/DOCX/TXT loaders and OCR fallback.",
                "Vietnamese Unicode normalization and paragraph reconstruction.",
                "Heading-aware semantic chunking with source metadata.",
                "Extractive and abstractive summarization compared with research metrics.",
            ],
            "findings": takeaways,
            "limitations": [
                "ROUGE/BLEU cần reference summary để phản ánh đúng chất lượng.",
                "Abstractive summaries cần citation grounding để giảm hallucination.",
            ],
        }

    @staticmethod
    def _quiz(takeaways: list[str], keywords: list[dict[str, Any]]) -> list[dict[str, Any]]:
        quiz = []
        terms = [item["term"] for item in keywords[:8]]
        for idx, takeaway in enumerate(takeaways[:6], start=1):
            answer = terms[(idx - 1) % len(terms)] if terms else "nội dung chính"
            quiz.append(
                {
                    "id": idx,
                    "type": "short_answer" if idx % 2 else "multiple_choice",
                    "difficulty": "medium" if idx <= 3 else "hard",
                    "question": f"Ý chính nào được thể hiện trong nhận định: {takeaway[:140]}?",
                    "answer": answer,
                    "options": terms[:4] if idx % 2 == 0 else [],
                }
            )
        return quiz

    @staticmethod
    def _flashcards(takeaways: list[str], keywords: list[dict[str, Any]]) -> list[dict[str, str]]:
        cards = [{"question": f"Khái niệm/từ khóa '{item['term']}' xuất hiện với vai trò gì?", "answer": "Liên quan trực tiếp đến chủ đề trung tâm của tài liệu."} for item in keywords[:8]]
        cards.extend({"question": "Nhận định chính cần ghi nhớ là gì?", "answer": takeaway} for takeaway in takeaways[:4])
        return cards[:12]

    @staticmethod
    def _mindmap(payload: dict[str, Any], keywords: list[dict[str, Any]]) -> dict[str, Any]:
        sections = payload.get("structure", {}).get("sections") or []
        root = payload.get("metadata", {}).get("title") or "Document"
        nodes = [{"id": "root", "label": root, "type": "document"}]
        edges = []
        for idx, section in enumerate(sections[:20], start=1):
            node_id = f"section-{idx}"
            nodes.append({"id": node_id, "label": section.get("title"), "type": "section"})
            edges.append({"source": "root", "target": node_id})
        for idx, keyword in enumerate(keywords[:12], start=1):
            node_id = f"keyword-{idx}"
            nodes.append({"id": node_id, "label": keyword["term"], "type": "keyword", "weight": keyword["weight"]})
            edges.append({"source": "root", "target": node_id})
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _presentation(payload: dict[str, Any], takeaways: list[str]) -> list[dict[str, Any]]:
        title = payload.get("metadata", {}).get("title") or "Document Intelligence Report"
        return [
            {"slide": 1, "title": title, "bullets": ["Mục tiêu phân tích", "Nguồn tài liệu", "Phương pháp NLP"]},
            {"slide": 2, "title": "Tổng quan tài liệu", "bullets": takeaways[:3]},
            {"slide": 3, "title": "Kết quả tóm tắt", "bullets": ["So sánh extractive vs abstractive", "Đánh giá factual consistency", "Citation grounding"]},
            {"slide": 4, "title": "Kết luận nghiên cứu", "bullets": takeaways[3:6] or takeaways[:3]},
        ]

    @staticmethod
    def _podcast_script(payload: dict[str, Any], takeaways: list[str]) -> dict[str, Any]:
        title = payload.get("metadata", {}).get("title") or "tài liệu"
        turns = [{"speaker": "Host A", "text": f"Hôm nay chúng ta phân tích {title} bằng hệ thống AI Document Intelligence."}]
        for idx, item in enumerate(takeaways[:6], start=1):
            turns.append({"speaker": "Host B" if idx % 2 else "Host A", "text": item})
        turns.append({"speaker": "Host A", "text": "Điểm quan trọng là mỗi nhận định đều cần được đối chiếu lại với citation nguồn."})
        return {"title": f"Podcast phân tích {title}", "tts_ready": True, "turns": turns}

    @staticmethod
    def _infographic(payload: dict[str, Any], keywords: list[dict[str, Any]], entities: list[dict[str, Any]]) -> dict[str, Any]:
        quality = payload.get("quality", {})
        return {
            "stats": [
                {"label": "Words", "value": quality.get("extraction", {}).get("word_count", 0)},
                {"label": "Chunks", "value": quality.get("chunk_count", len(payload.get("chunks") or []))},
                {"label": "Quality", "value": quality.get("extraction", {}).get("score", 0)},
                {"label": "Entities", "value": len(entities)},
            ],
            "top_keywords": keywords[:8],
            "entity_distribution": Counter(entity["type"] for entity in entities),
        }

    @staticmethod
    def _entity_graph(entities: list[dict[str, Any]], keywords: list[dict[str, Any]]) -> dict[str, Any]:
        nodes = []
        edges = []
        for idx, entity in enumerate(entities[:30], start=1):
            node_id = f"entity-{idx}"
            nodes.append({"id": node_id, "label": entity["text"], "type": entity["type"]})
            if idx > 1:
                edges.append({"source": "entity-1", "target": node_id, "weight": 0.4})
        for idx, keyword in enumerate(keywords[:8], start=1):
            node_id = f"concept-{idx}"
            nodes.append({"id": node_id, "label": keyword["term"], "type": "concept"})
            if nodes:
                edges.append({"source": nodes[0]["id"], "target": node_id, "weight": keyword["weight"]})
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _chunk_graph(chunks: list[dict[str, Any]]) -> dict[str, Any]:
        nodes = [
            {
                "id": chunk.get("chunk_id"),
                "label": f"Chunk {idx + 1}",
                "tokens": chunk.get("token_count"),
                "page": chunk.get("page_start"),
                "section": " / ".join(chunk.get("section_path") or []),
            }
            for idx, chunk in enumerate(chunks)
        ]
        edges = [
            {
                "source": chunks[idx].get("chunk_id"),
                "target": chunks[idx + 1].get("chunk_id"),
                "type": "sequence",
                "overlap": chunks[idx + 1].get("overlap_from_previous", False),
            }
            for idx in range(max(0, len(chunks) - 1))
        ]
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _embedding_map(chunks: list[dict[str, Any]], embeddings: list[list[float]] | None) -> list[dict[str, Any]]:
        if not embeddings:
            return []
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            return []
        centered = vectors - vectors.mean(axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            coords = centered @ vt[:2].T
        except Exception:
            coords = centered[:, :2] if centered.shape[1] >= 2 else np.pad(centered, ((0, 0), (0, 1)))
        if coords.ndim != 2:
            coords = coords.reshape(-1, 1)
        if coords.shape[1] < 2:
            coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
        max_abs = np.max(np.abs(coords)) or 1.0
        coords = coords / max_abs
        return [
            {
                "chunk_id": chunks[idx].get("chunk_id") if idx < len(chunks) else str(idx),
                "x": round(float(coords[idx, 0]), 4),
                "y": round(float(coords[idx, 1]), 4),
                "section": " / ".join((chunks[idx].get("section_path") if idx < len(chunks) else []) or []),
            }
            for idx in range(coords.shape[0])
        ]

    @staticmethod
    def _similarity_heatmap(embeddings: list[list[float]] | None, max_items: int = 24) -> list[list[float]]:
        if not embeddings:
            return []
        vectors = np.asarray(embeddings[:max_items], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            return []
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sims = (vectors / norms) @ (vectors / norms).T
        return np.round(sims, 4).tolist()

    @staticmethod
    def _chunk_hierarchy(chunks: list[dict[str, Any]]) -> dict[str, Any]:
        tree: dict[str, Any] = {"name": "Document", "children": []}
        section_nodes: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            section = " / ".join(chunk.get("section_path") or ["Unsectioned"])
            if section not in section_nodes:
                node = {"name": section, "children": []}
                section_nodes[section] = node
                tree["children"].append(node)
            section_nodes[section]["children"].append(
                {
                    "name": chunk.get("chunk_id"),
                    "tokens": chunk.get("token_count"),
                    "page": chunk.get("page_start"),
                }
            )
        return tree

    @staticmethod
    def _research_matrix(results: list[dict[str, Any]]) -> dict[str, Any]:
        from src.evaluate import aggregate_metric_rows

        rows = []
        for row in results:
            metrics = row.get("metrics", {})
            rows.append(
                {
                    "algorithm": row.get("algorithm"),
                    "group": row.get("group"),
                    "factual_risk": "low" if row.get("group") == "extractive" else "medium",
                    "rougeL": metrics.get("rougeL", 0.0),
                    "bertscore_f1": metrics.get("bertscore_f1", 0.0),
                    "semantic_similarity": metrics.get("semantic_similarity", 0.0),
                    "compression_ratio": metrics.get("compression_ratio", 0.0),
                    "latency": metrics.get("processing_time", 0.0),
                    "consistency": row.get("consistency", {}).get("consistency_score", 0.0),
                }
            )
        return {"rows": rows, "aggregate": aggregate_metric_rows([row.get("metrics", {}) for row in results])}


service = DocumentIntelligenceService()
