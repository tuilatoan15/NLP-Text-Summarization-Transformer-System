"""
raptor.py — RAPTOR-lite (Recursive Abstractive Processing for Tree-Organized Retrieval).

Thực hiện gom cụm các chunk cơ bản (K-Means), tóm tắt từng cụm bằng LLM và 
chèn các nút tóm tắt (Level 1) ngược lại vào Vector DB và Repository.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any
import numpy as np

from .summarizer import RAGTransformerSummarizer
from .rag_config import SUMMARIZE_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


def cluster_chunks(vectors: list[list[float]], num_clusters: int | None = None) -> list[list[int]]:
    """
    Gom cụm các vector chunk cơ bản sử dụng thuật toán K-Means thuần Numpy.
    Mặc định số cụm k = max(2, n_samples // 5).
    """
    X = np.array(vectors, dtype=np.float32)
    n_samples = X.shape[0]
    
    if n_samples <= 3:
        return [[i] for i in range(n_samples)]
        
    k = num_clusters or max(2, n_samples // 5)
    k = min(k, n_samples)
    
    # Khởi tạo centroids ngẫu nhiên từ các điểm dữ liệu
    rng = np.random.default_rng(42)
    centroids = X[rng.choice(n_samples, k, replace=False)]
    
    # Chạy K-Means tối đa 10 vòng lặp
    for _ in range(10):
        # Chuẩn hóa cosine distance
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        c_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
        
        similarities = X_norm @ c_norm.T  # Shape: (n_samples, k)
        labels = np.argmax(similarities, axis=1)
        
        new_centroids = []
        for i in range(k):
            members = X[labels == i]
            if len(members) > 0:
                new_centroids.append(members.mean(axis=0))
            else:
                new_centroids.append(X[rng.choice(n_samples)])
        centroids = np.array(new_centroids)
        
    clusters: list[list[int]] = [[] for _ in range(k)]
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
        
    return [c for c in clusters if c]


class RaptorIndexer:
    """
    Xây dựng cây phân cấp tóm tắt (RAPTOR-lite Tree) cho tài liệu.
    """

    def __init__(self, repository: Any, vector_store: Any, embedding_service: Any, generator: Any) -> None:
        self.repository = repository
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.generator = generator

    def build_tree(
        self,
        document_id: str,
        base_chunks: list[dict[str, Any]],
        base_vectors: list[list[float]],
        embedding_model: str,
    ) -> None:
        """
        Gom cụm các base chunks, sinh tóm tắt trừu tượng cho mỗi cụm (Level 1 Nodes),
        nhúng vector và lưu trữ chúng vào DB + Vector Index.
        """
        if len(base_chunks) <= 2:
            logger.info("📄 Tài liệu quá ngắn (<= 2 chunks), bỏ qua lập chỉ mục RAPTOR phân cấp.")
            return

        logger.info("🌲 Bắt đầu dựng cây RAPTOR-lite cho tài liệu %s...", document_id)
        
        # Gom cụm
        clusters = cluster_chunks(base_vectors)
        logger.info("⚡ Gom thành công %d cụm từ %d chunks.", len(clusters), len(base_chunks))

        summary_chunks = []
        filename = base_chunks[0]["filename"]

        for idx, cluster_indices in enumerate(clusters):
            # Ghép nội dung trong cụm
            cluster_text = "\n\n".join(base_chunks[i]["text"] for i in cluster_indices)
            child_ids = [base_chunks[i]["id"] for i in cluster_indices]
            
            # Sinh tóm tắt cho cụm bằng Generator có sẵn
            summary_text = self._summarize_cluster(cluster_text)
            if not summary_text:
                continue

            summary_id = f"raptor_{document_id}_{idx}_{str(uuid.uuid4())[:8]}"
            
            # Đóng gói nút tóm tắt mới
            summary_chunks.append({
                "id": summary_id,
                "document_id": document_id,
                "filename": filename,
                "page": None,  # Nút tóm tắt không thuộc trang cụ thể nào
                "chunk_index": -(idx + 1),  # Chỉ số âm để phân biệt với base chunks
                "text": f"[Tóm tắt phân đoạn tài liệu - {filename}]: {summary_text}",
                "metadata": {
                    "chunk_type": "summary",
                    "level": 1,
                    "child_chunk_ids": child_ids
                }
            })

        if not summary_chunks:
            logger.warning("⚠️ Không tạo được nút tóm tắt nào cho cây RAPTOR.")
            return

        # Nhúng vector cho các nút tóm tắt
        summary_vectors = self.embedding_service.embed_documents(
            [c["text"] for c in summary_chunks], embedding_model
        )

        # Lưu trữ
        self.repository.save_chunks(summary_chunks, summary_vectors, embedding_model)
        self.vector_store.upsert_chunks(summary_chunks, summary_vectors)
        
        logger.info(
            "🌲 Hoàn tất dựng cây RAPTOR-lite cho tài liệu %s: Đã thêm %d nút tóm tắt phân cấp.",
            document_id,
            len(summary_chunks)
        )

    def _summarize_cluster(self, text: str) -> str:
        """Sinh tóm tắt trừu tượng ngắn gọn bằng RAG Generator."""
        try:
            # Sử dụng prompt tóm tắt cô đọng
            prompt = (
                "Bạn là trợ lý phân tích văn bản. Hãy viết một đoạn tóm tắt ngắn gọn, mạch lạc khoảng 3-5 câu "
                "bao quát toàn bộ các thông tin cốt lõi trong đoạn văn bản sau, không tự bịa đặt thông tin:\n\n"
                f"{text}\n\n"
                "Bản tóm tắt tiếng Việt cô đọng:"
            )
            
            # Gọi trực tiếp qua summarizer để tận dụng cấu hình tối ưu
            from .rag_config import RAG_GENERATOR_TYPE
            
            summary = ""
            if RAG_GENERATOR_TYPE in {"gemini", "openai", "ollama"}:
                # Gọi API
                from .summarizer import _run_llm_api
                summary = _run_llm_api(prompt, RAG_GENERATOR_TYPE)
                
            if not summary:
                # Fallback sang local model
                from .summarizer import _pick_available_model, _run_transformer_generate, GENERATION_PROFILES
                model_key = _pick_available_model()
                if model_key:
                    profile = GENERATION_PROFILES[model_key]
                    summary = _run_transformer_generate(model_key, text, profile)
                    
            if not summary:
                # Extractive fallback tối giản
                sentences = text.split(".")
                summary = ". ".join(s.strip() for s in sentences[:3] if s.strip()) + "."

            return summary.strip()
        except Exception as exc:
            logger.error("Failed to summarize cluster: %s", exc)
            return ""
