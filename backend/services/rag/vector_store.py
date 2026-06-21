from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from src import config

logger = logging.getLogger(__name__)

# Try to import Qdrant and Chroma optional dependencies
try:
    import qdrant_client
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
except Exception:
    qdrant_client = None

try:
    import chromadb
    from chromadb.api.models.Collection import Collection
except Exception:
    chromadb = None
    Collection = Any


class VectorStoreManager:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._in_memory: dict[str, dict[str, Any]] = {}
        
        # Connection Clients
        self.qdrant_client_inst = None
        self.chroma_collection = None
        self.active_backend = "in_memory"

        # 1. Try to connect to Qdrant if selected
        if config.VECTOR_BACKEND == "qdrant" and qdrant_client is not None:
            try:
                host = config.QDRANT_HOST
                port = config.QDRANT_PORT
                logger.info(f"🔄 Connecting to Qdrant Server at {host}:{port} ...")
                
                # Connect to server
                self.qdrant_client_inst = qdrant_client.QdrantClient(host=host, port=port, timeout=3.0)
                
                # Check connection
                self.qdrant_client_inst.get_collections()
                self.active_backend = "qdrant"
                logger.info("✅ Successfully connected to Qdrant server backend.")
            except Exception as exc:
                logger.warning(f"⚠️ Could not connect to Qdrant server: {exc}. Trying Qdrant local persistent client...")
                try:
                    self.qdrant_client_inst = qdrant_client.QdrantClient(path=str(self.base_dir / "qdrant_local"))
                    self.active_backend = "qdrant"
                    logger.info("✅ Qdrant local persistent client initialized successfully.")
                except Exception as inner_exc:
                    logger.warning(f"⚠️ Qdrant local initialization failed: {inner_exc}. Will fallback to Chroma.")

        # 2. Fallback to ChromaDB if Qdrant is unavailable
        if self.active_backend == "in_memory" and chromadb is not None:
            try:
                logger.info("🔄 Falling back to ChromaDB ...")
                self.chroma_client = chromadb.PersistentClient(path=str(self.base_dir / "chroma"))
                self.chroma_collection = self.chroma_client.get_or_create_collection(
                    name="rag_chunks",
                    metadata={"hnsw:space": "cosine"},
                )
                self.active_backend = "chroma"
                logger.info("✅ ChromaDB persistent client fallback initialized.")
            except Exception as exc:
                logger.warning(f"⚠️ ChromaDB initialization failed: {exc}. Using in-memory fallback.")

        logger.info(f"🎯 Vector Store Active Backend: {self.active_backend.upper()}")

    def _ensure_qdrant_collection(self, dimension: int) -> None:
        """Create Qdrant collection if not exists."""
        if self.qdrant_client_inst is None:
            return
        
        try:
            exists = self.qdrant_client_inst.collection_exists(collection_name="rag_chunks")
            if not exists:
                logger.info(f"Creating new Qdrant collection 'rag_chunks' with dimension={dimension} ...")
                self.qdrant_client_inst.create_collection(
                    collection_name="rag_chunks",
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
                )
        except Exception as exc:
            logger.error(f"Failed to ensure Qdrant collection: {exc}")

    def upsert_chunks(self, chunks: list[dict[str, Any]], vectors: list[list[float]]) -> None:
        if not chunks:
            return

        # --- A. QDRANT BACKEND ---
        if self.active_backend == "qdrant" and self.qdrant_client_inst is not None:
            try:
                dimension = len(vectors[0])
                self._ensure_qdrant_collection(dimension)

                points = []
                for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
                    # Check dimension match
                    if len(vector) != dimension:
                        logger.warning("Dimension mismatch skipped.")
                        continue
                    
                    payload = {
                        "chunk_id": str(chunk["id"]),
                        "document_id": str(chunk["document_id"]),
                        "filename": str(chunk["filename"]),
                        "page": int(chunk.get("page") or -1),
                        "chunk_index": int(chunk["chunk_index"]),
                        "text": str(chunk["text"]),
                        "chunk_type": str(chunk.get("metadata", {}).get("chunk_type", "base")),
                        "level": int(chunk.get("metadata", {}).get("level", 0)),
                    }
                    
                    # Convert string UUID or custom ID to valid integer or string format for Qdrant
                    point_id = chunk["id"]
                    try:
                        import uuid
                        point_uuid = str(uuid.UUID(point_id))
                    except ValueError:
                        import uuid
                        point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id))
                    
                    points.append(PointStruct(id=point_uuid, vector=vector, payload=payload))

                if points:
                    self.qdrant_client_inst.upsert(
                        collection_name="rag_chunks",
                        wait=True,
                        points=points
                    )
                    logger.info(f"✅ Qdrant: Upserted {len(points)} chunks successfully.")
                return
            except Exception as exc:
                logger.error(f"❌ Qdrant upsert failed: {exc}. Trying fallback to Chroma/In-Memory...")

        # --- B. CHROMADB BACKEND ---
        if self.active_backend == "chroma" and self.chroma_collection is not None:
            ids = [chunk["id"] for chunk in chunks]
            metadatas = [
                {
                    "document_id": chunk["document_id"],
                    "filename": chunk["filename"],
                    "page": chunk.get("page") or -1,
                    "chunk_index": chunk["chunk_index"],
                    "chunk_type": chunk.get("metadata", {}).get("chunk_type", "base"),
                    "level": chunk.get("metadata", {}).get("level", 0),
                }
                for chunk in chunks
            ]
            documents = [chunk["text"] for chunk in chunks]
            try:
                self.chroma_collection.upsert(ids=ids, embeddings=vectors, metadatas=metadatas, documents=documents)
                logger.info(f"✅ Chroma: Upserted {len(chunks)} chunks.")
                return
            except Exception as exc:
                logger.error(f"❌ Chroma upsert failed: {exc}")

        # --- C. IN-MEMORY FALLBACK ---
        logger.warning("⚠️ Using in-memory fallback for upsert.")
        for chunk, vector in zip(chunks, vectors):
            self._in_memory[chunk["id"]] = {
                **chunk,
                "vector": vector,
                "embedding_score": 0.0,
            }

    def query(
        self,
        *,
        query_vector: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        # --- A. QDRANT BACKEND ---
        if self.active_backend == "qdrant" and self.qdrant_client_inst is not None:
            try:
                self._ensure_qdrant_collection(len(query_vector))
                
                # Setup metadata filter if document_ids are provided
                q_filter = None
                if document_ids:
                    conditions = [
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=doc_id)
                        )
                        for doc_id in document_ids
                    ]
                    # If multiple document ids, we match any
                    if len(conditions) == 1:
                        q_filter = Filter(must=conditions)
                    else:
                        q_filter = Filter(should=conditions)

                search_result = self.qdrant_client_inst.query_points(
                    collection_name="rag_chunks",
                    query=query_vector,
                    query_filter=q_filter,
                    limit=max(top_k, 1),
                    with_payload=True,
                    with_vectors=True
                )

                hits = []
                for hit in search_result.points:
                    payload = hit.payload or {}
                    # Cosine distance in Qdrant ranges from -1 to 1 or 0 to 1 based on configuration
                    # Hit.score is already cosine similarity [0, 1] if distance configured as COSINE
                    score = float(hit.score)
                    
                    hits.append({
                        "id": str(payload.get("chunk_id", hit.id)),
                        "document_id": str(payload.get("document_id", "")),
                        "filename": str(payload.get("filename", "")),
                        "page": None if int(payload.get("page", -1)) < 0 else int(payload.get("page", -1)),
                        "chunk_index": int(payload.get("chunk_index", 0)),
                        "text": str(payload.get("text", "")),
                        "vector": hit.vector or query_vector,
                        "embedding_score": score,
                        "metadata": {
                            "chunk_type": payload.get("chunk_type", "base"),
                            "level": int(payload.get("level", 0)),
                        }
                    })
                return hits
            except Exception as exc:
                logger.error(f"❌ Qdrant query failed: {exc}. Trying Chroma fallback...")

        # --- B. CHROMADB BACKEND ---
        if self.active_backend == "chroma" and self.chroma_collection is not None:
            where = {"document_id": {"$in": document_ids}} if document_ids else None
            try:
                result = self.chroma_collection.query(
                    query_embeddings=[query_vector],
                    n_results=max(top_k, 1),
                    where=where,
                    include=["distances", "documents", "metadatas", "embeddings"],
                )
                
                ids = result.get("ids", [[]])[0]
                distances = result.get("distances", [[]])[0]
                documents = result.get("documents", [[]])[0]
                metadatas = result.get("metadatas", [[]])[0]
                embeddings = result.get("embeddings", [[]])[0]

                hits = []
                for idx, chunk_id in enumerate(ids):
                    meta = metadatas[idx] or {}
                    distance = float(distances[idx]) if idx < len(distances) else 1.0
                    score = 1.0 - min(1.0, max(0.0, distance))
                    hits.append({
                        "id": chunk_id,
                        "document_id": str(meta.get("document_id", "")),
                        "filename": str(meta.get("filename", "")),
                        "page": None if int(meta.get("page", -1)) < 0 else int(meta.get("page", -1)),
                        "chunk_index": int(meta.get("chunk_index", 0)),
                        "text": documents[idx] if idx < len(documents) else "",
                        "vector": embeddings[idx] if idx < len(embeddings) else query_vector,
                        "embedding_score": score,
                        "metadata": {
                            "chunk_type": meta.get("chunk_type", "base"),
                            "level": int(meta.get("level", 0)),
                        }
                    })
                return hits
            except Exception as exc:
                logger.error(f"❌ Chroma query failed: {exc}")

        # --- C. IN-MEMORY FALLBACK ---
        rows = list(self._in_memory.values())
        if document_ids:
            rows = [r for r in rows if r.get("document_id") in document_ids]
        
        # Simple local cosine similarity calculation for fallback
        import numpy as np
        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q) or 1.0
        
        for r in rows:
            vec = np.array(r.get("vector", []), dtype=np.float32)
            if vec.size == 0:
                sim = 0.0
            else:
                sim = float(np.dot(q, vec) / ((np.linalg.norm(vec) or 1.0) * q_norm))
            r["embedding_score"] = sim
            
        rows.sort(key=lambda x: x["embedding_score"], reverse=True)
        return rows[:top_k]

    def delete_document(self, document_id: str) -> None:
        # --- A. QDRANT BACKEND ---
        if self.active_backend == "qdrant" and self.qdrant_client_inst is not None:
            try:
                self.qdrant_client_inst.delete(
                    collection_name="rag_chunks",
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=document_id)
                            )
                        ]
                    )
                )
                logger.info(f"Qdrant: Deleted chunks for document {document_id}")
                return
            except Exception as exc:
                logger.error(f"Qdrant delete failed: {exc}")

        # --- B. CHROMADB BACKEND ---
        if self.active_backend == "chroma" and self.chroma_collection is not None:
            self.chroma_collection.delete(where={"document_id": document_id})
            logger.info(f"Chroma: Deleted chunks for document {document_id}")
            return

        # --- C. IN-MEMORY FALLBACK ---
        self._in_memory = {
            k: v for k, v in self._in_memory.items() if v.get("document_id") != document_id
        }

    def delete_all_documents(self) -> None:
        """Xóa toàn bộ vectors khỏi tất cả backends."""
        # --- A. QDRANT BACKEND ---
        if self.active_backend == "qdrant" and self.qdrant_client_inst is not None:
            try:
                self.qdrant_client_inst.delete_collection("rag_chunks")
                logger.info("Qdrant: Deleted entire rag_chunks collection")
            except Exception as exc:
                logger.error(f"Qdrant delete_all failed: {exc}")

        # --- B. CHROMADB BACKEND ---
        if self.active_backend == "chroma" and self.chroma_collection is not None:
            try:
                self.chroma_collection.delete()
                logger.info("Chroma: Deleted all chunks")
            except Exception as exc:
                logger.error(f"Chroma delete_all failed: {exc}")

        # --- C. IN-MEMORY FALLBACK ---
        self._in_memory.clear()
