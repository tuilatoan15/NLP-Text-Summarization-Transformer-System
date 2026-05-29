from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import chromadb
    from chromadb.api.models.Collection import Collection
except Exception:  # pragma: no cover - optional dependency fallback
    chromadb = None
    Collection = Any  # type: ignore[assignment]


class VectorStoreManager:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._in_memory: dict[str, dict[str, Any]] = {}
        self.collection: Collection | None = None
        if chromadb is not None:
            self.client = chromadb.PersistentClient(path=str(self.base_dir / "chroma"))
            self.collection = self.client.get_or_create_collection(
                name="rag_chunks",
                metadata={"hnsw:space": "cosine"},
            )

    def upsert_chunks(self, chunks: list[dict[str, Any]], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        if self.collection is None:
            for chunk, vector in zip(chunks, vectors):
                self._in_memory[chunk["id"]] = {
                    **chunk,
                    "vector": vector,
                    "embedding_score": 0.0,
                }
            return
        ids = [chunk["id"] for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk["document_id"],
                "filename": chunk["filename"],
                "page": chunk.get("page") or -1,
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ]
        documents = [chunk["text"] for chunk in chunks]
        try:
            self.collection.upsert(ids=ids, embeddings=vectors, metadatas=metadatas, documents=documents)
        except Exception as exc:
            if "dimension" in str(exc).lower() or "expecting embedding" in str(exc).lower():
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("Dimension mismatch detected in ChromaDB collection during upsert. Recreating collection...")
                try:
                    self.client.delete_collection("rag_chunks")
                    self.collection = self.client.create_collection(
                        name="rag_chunks",
                        metadata={"hnsw:space": "cosine"},
                    )
                    self.collection.upsert(ids=ids, embeddings=vectors, metadatas=metadatas, documents=documents)
                    logger.info("ChromaDB collection successfully recreated with new dimension.")
                    return
                except Exception as inner_exc:
                    logger.error("Failed to recreate ChromaDB collection on dimension mismatch: %s", inner_exc)
                    raise inner_exc
            raise exc

    def query(
        self,
        *,
        query_vector: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if self.collection is None:
            rows = list(self._in_memory.values())
            if document_ids:
                rows = [r for r in rows if r.get("document_id") in document_ids]
            return rows[:top_k]

        where = {"document_id": {"$in": document_ids}} if document_ids else None
        try:
            result = self.collection.query(
                query_embeddings=[query_vector],
                n_results=max(top_k, 1),
                where=where,
                include=["distances", "documents", "metadatas", "embeddings"],
            )
        except Exception as exc:
            if "dimension" in str(exc).lower() or "expecting embedding" in str(exc).lower():
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("Dimension mismatch detected during ChromaDB query. Recreating collection...")
                try:
                    self.client.delete_collection("rag_chunks")
                    self.collection = self.client.create_collection(
                        name="rag_chunks",
                        metadata={"hnsw:space": "cosine"},
                    )
                    return []
                except Exception as inner_exc:
                    logger.error("Failed to recreate ChromaDB collection during query: %s", inner_exc)
                    raise inner_exc
            raise exc

        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        embeddings = result.get("embeddings", [[]])[0]

        hits: list[dict[str, Any]] = []
        for idx, chunk_id in enumerate(ids):
            meta = metadatas[idx] or {}
            distance = float(distances[idx]) if idx < len(distances) else 1.0
            score = 1.0 - min(1.0, max(0.0, distance))
            hits.append(
                {
                    "id": chunk_id,
                    "document_id": str(meta.get("document_id", "")),
                    "filename": str(meta.get("filename", "")),
                    "page": None if int(meta.get("page", -1)) < 0 else int(meta.get("page", -1)),
                    "chunk_index": int(meta.get("chunk_index", 0)),
                    "text": documents[idx] if idx < len(documents) else "",
                    "vector": embeddings[idx] if idx < len(embeddings) else query_vector,
                    "embedding_score": score,
                }
            )
        return hits

    def delete_document(self, document_id: str) -> None:
        if self.collection is None:
            self._in_memory = {
                k: v for k, v in self._in_memory.items() if v.get("document_id") != document_id
            }
            return
        self.collection.delete(where={"document_id": document_id})

