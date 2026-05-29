import sys
sys.path.insert(0, ".")

import logging
from pathlib import Path
from backend.services.rag.repository import RAGRepository
from backend.services.rag.vector_store import VectorStoreManager
from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sync_db")

db_path = config.STORAGE_DIR / "document_intelligence" / "rag" / "rag_chat.db"
rag_dir = config.STORAGE_DIR / "document_intelligence" / "rag"

logger.info("Initializing RAG Repository...")
repository = RAGRepository(db_path)

logger.info("Initializing VectorStoreManager...")
vector_store = VectorStoreManager(rag_dir)

if vector_store.collection is None:
    logger.error("ChromaDB is still None! Ensure chromadb is installed and imported correctly.")
    sys.exit(1)

# List all chunks from SQLite
logger.info("Reading chunks and vectors from SQLite...")
sqlite_chunks = repository.list_chunks()
logger.info(f"Found {len(sqlite_chunks)} chunks in SQLite.")

# Check count in ChromaDB
chroma_count = vector_store.collection.count()
logger.info(f"Found {chroma_count} chunks in ChromaDB.")

if len(sqlite_chunks) > 0:
    logger.info("Syncing SQLite chunks to ChromaDB...")
    chunks_to_upsert = []
    vectors_to_upsert = []
    
    for c in sqlite_chunks:
        # Prepare chunk format for upsert_chunks
        # vector_store.upsert_chunks expects a list of chunk dicts and a list of vectors
        chunks_to_upsert.append({
            "id": c["id"],
            "document_id": c["document_id"],
            "filename": c["filename"],
            "page": c["page"],
            "chunk_index": c["chunk_index"],
            "text": c["text"],
            "metadata": c["metadata"]
        })
        vectors_to_upsert.append(c["vector"])
        
    logger.info(f"Upserting {len(chunks_to_upsert)} chunks to ChromaDB...")
    try:
        vector_store.upsert_chunks(chunks_to_upsert, vectors_to_upsert)
        new_count = vector_store.collection.count()
        logger.info(f"Sync complete! New ChromaDB chunk count: {new_count}")
    except Exception as e:
        logger.error(f"Failed to upsert chunks: {e}")
else:
    logger.info("No chunks in SQLite to sync.")
