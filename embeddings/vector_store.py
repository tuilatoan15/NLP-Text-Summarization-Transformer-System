"""Vector store abstraction: local JSON, FAISS, or ChromaDB."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src import config
from src.utils import logger


class VectorStore(ABC):
    @abstractmethod
    def upsert(
        self,
        document_id: str,
        chunk_ids: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        ...

    @abstractmethod
    def search(self, document_id: str, query_vector: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        ...


class LocalVectorStore(VectorStore):
    """Persist vectors beside document JSON (development default)."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or config.DOCUMENT_INTELLIGENCE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def upsert(
        self,
        document_id: str,
        chunk_ids: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        payload = {
            "document_id": document_id,
            "chunk_ids": chunk_ids,
            "embeddings": embeddings.tolist(),
            "metadatas": metadatas or [],
        }
        path = self.base_dir / f"{document_id}.vectors.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def search(self, document_id: str, query_vector: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        path = self.base_dir / f"{document_id}.vectors.json"
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        vectors = np.asarray(payload["embeddings"], dtype=np.float32)
        if vectors.ndim != 2:
            return []
        q = query_vector.astype(np.float32)
        q_norm = np.linalg.norm(q) or 1.0
        v_norms = np.linalg.norm(vectors, axis=1)
        v_norms[v_norms == 0] = 1.0
        scores = (vectors @ q) / (v_norms * q_norm)
        order = np.argsort(-scores)[:top_k]
        return [
            {
                "chunk_id": payload["chunk_ids"][int(i)],
                "score": round(float(scores[int(i)]), 4),
                "metadata": (payload.get("metadatas") or [{}])[int(i)] if payload.get("metadatas") else {},
            }
            for i in order
        ]


class FaissVectorStore(VectorStore):
    def __init__(self, base_dir: Path | None = None) -> None:
        import faiss  # type: ignore

        self.faiss = faiss
        self.base_dir = base_dir or (config.CACHE_DIR / "faiss")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._indexes: dict[str, Any] = {}

    def _index_path(self, document_id: str) -> Path:
        return self.base_dir / f"{document_id}.faiss"

    def upsert(
        self,
        document_id: str,
        chunk_ids: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        dim = embeddings.shape[1]
        index = self.faiss.IndexFlatIP(dim)
        vectors = embeddings.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        index.add(vectors / norms)
        self.faiss.write_index(index, str(self._index_path(document_id)))
        meta_path = self.base_dir / f"{document_id}.meta.json"
        meta_path.write_text(
            json.dumps({"chunk_ids": chunk_ids, "metadatas": metadatas or []}, ensure_ascii=False),
            encoding="utf-8",
        )
        self._indexes[document_id] = index

    def search(self, document_id: str, query_vector: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        path = self._index_path(document_id)
        if not path.exists():
            return []
        index = self.faiss.read_index(str(path))
        meta = json.loads((self.base_dir / f"{document_id}.meta.json").read_text(encoding="utf-8"))
        q = query_vector.astype(np.float32).reshape(1, -1)
        q_norm = np.linalg.norm(q) or 1.0
        scores, indices = index.search(q / q_norm, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                {
                    "chunk_id": meta["chunk_ids"][int(idx)],
                    "score": round(float(score), 4),
                    "metadata": (meta.get("metadatas") or [{}])[int(idx)],
                }
            )
        return results


class ChromaVectorStore(VectorStore):
    def __init__(self) -> None:
        import chromadb  # type: ignore

        host = config.CHROMA_HOST
        port = config.CHROMA_PORT
        self.client = chromadb.HttpClient(host=host, port=port)

    def _collection(self, document_id: str):
        return self.client.get_or_create_collection(name=f"doc_{document_id}")

    def upsert(
        self,
        document_id: str,
        chunk_ids: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        col = self._collection(document_id)
        col.upsert(
            ids=chunk_ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas or [{} for _ in chunk_ids],
        )

    def search(self, document_id: str, query_vector: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        col = self._collection(document_id)
        result = col.query(query_embeddings=[query_vector.tolist()], n_results=top_k)
        items = []
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        for chunk_id, dist, meta in zip(ids, distances, metadatas):
            items.append({"chunk_id": chunk_id, "score": round(1.0 - float(dist), 4), "metadata": meta or {}})
        return items


def get_vector_store() -> VectorStore:
    backend = config.VECTOR_BACKEND
    if backend == "faiss":
        try:
            return FaissVectorStore()
        except ImportError:
            logger.warning("faiss-cpu not installed; falling back to local vector store")
    if backend == "chroma":
        try:
            return ChromaVectorStore()
        except Exception as exc:
            logger.warning("Chroma unavailable (%s); falling back to local", exc)
    return LocalVectorStore()


@dataclass(slots=True)
class SearchHit:
    index: int
    score: float
    metadata: dict[str, Any] | None = None


class VectorIndex:
    """In-memory cosine index for a single document (FAISS when available)."""

    def __init__(
        self,
        embeddings: np.ndarray,
        *,
        ids: list[str] | None = None,
        normalize: bool = True,
    ) -> None:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("embeddings must be a 2-D matrix")
        if normalize and vectors.size:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-8)
        self._vectors = vectors
        self.ids = ids or [str(i) for i in range(len(vectors))]
        self._faiss_index = None
        self._backend = "numpy"
        self._build_faiss()

    @property
    def dimension(self) -> int:
        return int(self._vectors.shape[1]) if self._vectors.size else 0

    @property
    def backend(self) -> str:
        return self._backend

    def _build_faiss(self) -> None:
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(self.dimension)
            index.add(self._vectors)
            self._faiss_index = index
            self._backend = "faiss"
        except ImportError:
            self._faiss_index = None
            self._backend = "numpy"

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchHit]:
        if self._vectors.size == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        k = max(1, min(top_k, len(self.ids)))

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query, k)
            return [
                SearchHit(index=int(idx), score=float(scores[0][i]), metadata={"id": self.ids[int(idx)]})
                for i, idx in enumerate(indices[0])
                if int(idx) >= 0
            ]

        scores = (self._vectors @ query.T).reshape(-1)
        ranked = np.argsort(scores)[::-1][:k]
        return [
            SearchHit(index=int(idx), score=float(scores[idx]), metadata={"id": self.ids[int(idx)]})
            for idx in ranked
        ]
