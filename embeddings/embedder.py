"""High-quality multilingual embedding wrapper with GPU and low-VRAM safeguards."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from pipeline.schema import EmbeddingConfig
from utils.logger import logger


class EmbeddingModelRegistry:
    """Recommended embedding models for Vietnamese summarization/RAG."""

    MODELS: dict[str, dict[str, str | int | bool]] = {
        "BAAI/bge-m3": {
            "family": "BGE",
            "dimension": 1024,
            "strength": "Strong multilingual retrieval, long context, good Vietnamese support.",
            "passage_prefix": "",
            "query_prefix": "",
        },
        "intfloat/multilingual-e5-large": {
            "family": "E5",
            "dimension": 1024,
            "strength": "High semantic quality for multilingual retrieval; requires query/passsage prefixes.",
            "passage_prefix": "passage: ",
            "query_prefix": "query: ",
        },
        "jinaai/jina-embeddings-v3": {
            "family": "Jina",
            "dimension": 1024,
            "strength": "Modern multilingual embedding with task-specific adapters and long context.",
            "passage_prefix": "",
            "query_prefix": "",
            "trust_remote_code": True,
        },
        "keepitreal/vietnamese-sbert": {
            "family": "Vietnamese SBERT",
            "dimension": 768,
            "strength": "Lightweight Vietnamese semantic similarity baseline.",
            "passage_prefix": "",
            "query_prefix": "",
        },
        "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base": {
            "family": "PhoBERT SimCSE",
            "dimension": 768,
            "strength": "Vietnamese-specific sentence similarity; useful baseline for local language quality.",
            "passage_prefix": "",
            "query_prefix": "",
        },
    }

    @classmethod
    def defaults_for(cls, model_name: str) -> dict[str, str | int | bool]:
        return dict(cls.MODELS.get(model_name, {}))

    @classmethod
    def list_models(cls) -> dict[str, dict[str, str | int | bool]]:
        return dict(cls.MODELS)


@dataclass(slots=True)
class EmbeddingBatchResult:
    embeddings: np.ndarray
    model_name: str
    dimension: int
    normalized: bool
    provider: str


class SentenceTransformerEmbedder:
    """SentenceTransformer embedder with deterministic fallback for tests/offline use."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        defaults = EmbeddingModelRegistry.defaults_for(self.config.model_name)
        self.query_prefix = self.config.query_prefix
        self.passage_prefix = self.config.passage_prefix
        if self.query_prefix is None:
            self.query_prefix = str(defaults.get("query_prefix", ""))
        if self.passage_prefix is None:
            self.passage_prefix = str(defaults.get("passage_prefix", ""))
        self._model = None
        self._provider = "sentence-transformers"

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def embed_documents(self, texts: Iterable[str]) -> EmbeddingBatchResult:
        prepared = [self._prepare_passage(text) for text in texts]
        return self._embed(prepared)

    def embed_query(self, text: str) -> np.ndarray:
        result = self._embed([self._prepare_query(text)])
        return result.embeddings[0]

    def _embed(self, texts: list[str]) -> EmbeddingBatchResult:
        if not texts:
            return EmbeddingBatchResult(
                embeddings=np.zeros((0, 0), dtype=np.float32),
                model_name=self.config.model_name,
                dimension=0,
                normalized=self.config.normalize_embeddings,
                provider=self._provider,
            )

        model = self._load_model()
        if model is None:
            vectors = self._hash_embeddings(texts)
            return EmbeddingBatchResult(
                embeddings=vectors,
                model_name="hash-fallback",
                dimension=vectors.shape[1],
                normalized=True,
                provider="hashing",
            )

        try:
            embeddings = model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=self.config.show_progress,
                convert_to_numpy=True,
            )
            vectors = np.asarray(embeddings, dtype=np.float32)
            return EmbeddingBatchResult(
                embeddings=vectors,
                model_name=self.config.model_name,
                dimension=vectors.shape[1] if vectors.ndim == 2 else 0,
                normalized=self.config.normalize_embeddings,
                provider=self._provider,
            )
        finally:
            self._clear_cuda_cache()

    def _load_model(self):
        if self.config.model_name.lower() in {"hash", "hash-fallback", "offline"}:
            return None
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            kwargs = {"trust_remote_code": self.config.trust_remote_code}
            if self.config.device:
                kwargs["device"] = self.config.device
            self._model = SentenceTransformer(self.config.model_name, **kwargs)
            try:
                self._model.max_seq_length = self.config.max_seq_length
            except Exception:
                pass
            if self.config.use_fp16:
                self._try_half_precision()
            return self._model
        except Exception as exc:
            if not self.config.fallback_to_hashing:
                raise
            logger.warning(
                "Embedding model %s unavailable. Using deterministic hash fallback: %s",
                self.config.model_name,
                exc,
            )
            return None

    def _try_half_precision(self) -> None:
        try:
            import torch

            if torch.cuda.is_available() and self._model is not None:
                self._model = self._model.half()
        except Exception as exc:
            logger.debug("Could not enable fp16 for embeddings: %s", exc)

    def _prepare_passage(self, text: str) -> str:
        text = text or ""
        return f"{self.passage_prefix}{text}" if self.passage_prefix else text

    def _prepare_query(self, text: str) -> str:
        text = text or ""
        return f"{self.query_prefix}{text}" if self.query_prefix else text

    @staticmethod
    def _clear_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _hash_embeddings(self, texts: list[str], dimension: int = 384) -> np.ndarray:
        vectors = np.zeros((len(texts), dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = text.lower().split()
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row, index] += sign
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return vectors
