from __future__ import annotations

import logging
import uuid
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


def gmm_cluster(X: np.ndarray, k: int, max_iter: int = 10, threshold: float = 0.15) -> list[list[int]]:
    """
    Phân cụm các vector sử dụng Gaussian Mixture Model (GMM) đường chéo bằng Numpy.
    Hỗ trợ phân cụm mềm chéo (overlapping clusters): một điểm có xác suất > threshold
    có thể thuộc nhiều cụm khác nhau.
    """
    n_samples, n_features = X.shape
    if n_samples <= k:
        return [[i] for i in range(n_samples)]

    # 1. Khởi tạo Means bằng các điểm ngẫu nhiên
    rng = np.random.default_rng(42)
    means = X[rng.choice(n_samples, k, replace=False)]

    # 2. Khởi tạo covariances (diagonal) và weights
    covars = np.ones((k, n_features), dtype=np.float32) * (X.var(axis=0) + 1e-5)
    weights = np.ones(k, dtype=np.float32) / k

    # 3. Vòng lặp Expectation-Maximization (EM)
    for _ in range(max_iter):
        # E-step: Tính responsibilities
        resp = np.zeros((n_samples, k), dtype=np.float32)
        for j in range(k):
            diff = X - means[j]
            var = covars[j]
            var[var == 0] = 1e-9
            exponent = -0.5 * np.sum((diff ** 2) / var, axis=1)
            log_det = np.sum(np.log(var))
            log_prob = exponent - 0.5 * (n_features * np.log(2 * np.pi) + log_det)
            resp[:, j] = weights[j] * np.exp(log_prob)

        row_sums = resp.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-9
        resp = resp / row_sums

        # M-step: Cập nhật parameters
        Nk = resp.sum(axis=0)
        Nk[Nk == 0] = 1e-9
        weights = Nk / n_samples
        means = (resp.T @ X) / Nk[:, np.newaxis]

        for j in range(k):
            diff = X - means[j]
            covars[j] = (resp[:, j] @ (diff ** 2)) / Nk[j]
            covars[j] = np.clip(covars[j], 1e-5, None)

    # 4. Gán các điểm dữ liệu vào các cụm dựa trên ngưỡng xác suất mềm
    clusters: list[list[int]] = [[] for _ in range(k)]
    for i in range(n_samples):
        assigned = False
        for j in range(k):
            if resp[i, j] > threshold:
                clusters[j].append(i)
                assigned = True
        if not assigned:
            clusters[np.argmax(resp[i])].append(i)

    return [c for c in clusters if c]


class RaptorIndexer:
    """
    Xây dựng cây phân cấp tóm tắt đệ quy (RAPTOR) phân cụm mềm cho tài liệu.
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
        max_levels: int = 3
    ) -> None:
        """
        Dựng cây RAPTOR phân cấp đệ quy:
        Base chunks (Level 0) -> GMM Clustered Summaries (Level 1) -> Level 2 -> Level 3.
        """
        if len(base_chunks) <= 2:
            logger.info("📄 Tài liệu quá ngắn (<= 2 chunks), bỏ qua lập chỉ mục RAPTOR phân cấp.")
            return

        logger.info("🌲 Bắt đầu dựng cây RAPTOR phân cấp đệ quy cho tài liệu %s...", document_id)

        current_chunks = base_chunks
        current_vectors = base_vectors
        level = 1
        filename = base_chunks[0]["filename"]

        while level <= max_levels:
            n_samples = len(current_chunks)
            if n_samples <= 3:
                logger.info("Stop recursive clustering at Level %d because node count is too small (%d)", level, n_samples)
                break

            # Xác định số cụm k hợp lý
            k = max(2, n_samples // 5)
            k = min(k, n_samples - 1)

            X = np.array(current_vectors, dtype=np.float32)
            clusters = gmm_cluster(X, k=k)
            logger.info("Level %d: Phân cụm %d nút thành %d cụm GMM.", level, n_samples, len(clusters))

            level_chunks = []
            for idx, cluster_indices in enumerate(clusters):
                cluster_text = "\n\n".join(current_chunks[i]["text"] for i in cluster_indices)
                child_ids = [current_chunks[i]["id"] for i in cluster_indices]

                summary_text = self._summarize_cluster(cluster_text)
                if not summary_text:
                    continue

                summary_id = f"raptor_{document_id}_L{level}_{idx}_{str(uuid.uuid4())[:8]}"
                level_chunks.append({
                    "id": summary_id,
                    "document_id": document_id,
                    "filename": filename,
                    "page": None,
                    "chunk_index": -(idx + 1) * (10 ** level),
                    "text": f"[Tóm tắt phân cấp L{level} - {filename}]: {summary_text}",
                    "metadata": {
                        "chunk_type": "summary",
                        "level": level,
                        "child_chunk_ids": child_ids
                    }
                })

            if not level_chunks:
                logger.info("Level %d: Không tạo thêm được nút tóm tắt nào, kết thúc đệ quy.", level)
                break

            # Nhúng các nút tóm tắt mới của Level này
            level_vectors = self.embedding_service.embed_documents(
                [c["text"] for c in level_chunks], embedding_model
            )

            # Lưu trữ
            self.repository.save_chunks(level_chunks, level_vectors, embedding_model)
            self.vector_store.upsert_chunks(level_chunks, level_vectors)

            # Chuẩn bị cho cấp tiếp theo
            current_chunks = level_chunks
            current_vectors = level_vectors
            level += 1

        logger.info("🌲 Hoàn tất dựng cây RAPTOR đệ quy cho tài liệu %s.", document_id)

    def _summarize_cluster(self, text: str) -> str:
        """Sinh tóm tắt trừu tượng ngắn gọn bằng RAG Generator."""
        try:
            prompt = (
                "Bạn là trợ lý phân tích văn bản. Hãy viết một đoạn tóm tắt ngắn gọn, mạch lạc khoảng 3-5 câu "
                "bao quát toàn bộ các thông tin cốt lõi trong đoạn văn bản sau, không tự bịa đặt thông tin:\n\n"
                f"{text}\n\n"
                "Bản tóm tắt tiếng Việt cô đọng:"
            )
            
            from .rag_config import RAG_GENERATOR_TYPE
            summary = ""
            if RAG_GENERATOR_TYPE in {"gemini", "openai", "ollama"}:
                from .summarizer import _run_llm_api
                summary = _run_llm_api(prompt, RAG_GENERATOR_TYPE)
                
            if not summary:
                from .summarizer import _pick_available_model, _run_transformer_generate
                from .rag_config import resolve_generation_profile
                model_key = _pick_available_model()
                if model_key:
                    profile = resolve_generation_profile(model_key)
                    summary = _run_transformer_generate(model_key, text, profile)
                    
            if not summary:
                sentences = text.split(".")
                summary = ". ".join(s.strip() for s in sentences[:3] if s.strip()) + "."

            return summary.strip()
        except Exception as exc:
            logger.error("Failed to summarize cluster: %s", exc)
            return ""
