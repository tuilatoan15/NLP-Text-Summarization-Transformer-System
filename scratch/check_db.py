import sys
sys.path.insert(0, ".")

import sqlite3
from pathlib import Path
from backend.services.rag.vector_store import VectorStoreManager
from backend.services.rag.rag_config import EMBEDDING_MODEL
from src import config

db_path = config.STORAGE_DIR / "document_intelligence" / "rag" / "rag_chat.db"
print("SQLite DB Path:", db_path)
print("File exists:", db_path.exists())

if db_path.exists():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in SQLite:", tables)
    
    # Check documents
    try:
        cursor.execute("SELECT id, filename, source_type, metadata FROM documents;")
        docs = cursor.fetchall()
        print(f"\nDocuments ({len(docs)}):")
        for d in docs:
            print(f" - ID: {d[0]}, Filename: {d[1]}, Type: {d[2]}, Metadata: {d[3]}")
    except Exception as e:
        print("Error reading documents:", e)
        
    # Check chunks
    try:
        cursor.execute("SELECT COUNT(*) FROM chunks;")
        chunk_count = cursor.fetchone()[0]
        print(f"Chunks in SQLite: {chunk_count}")
    except Exception as e:
        print("Error reading chunks:", e)
        
    conn.close()

print("\n--- ChromaDB ---")
vector_store = VectorStoreManager(config.STORAGE_DIR / "document_intelligence" / "rag")
if vector_store.collection is not None:
    try:
        col_count = vector_store.collection.count()
        print("ChromaDB Collection Name:", vector_store.collection.name)
        print("Chunks in ChromaDB collection:", col_count)
        
        # Peek at some metadata/embeddings
        if col_count > 0:
            peek = vector_store.collection.peek(limit=2)
            print("Peek keys:", peek.keys())
            if peek.get("metadatas"):
                print("Sample metadata:", peek["metadatas"][0])
            if peek.get("embeddings"):
                print("Embedding size:", len(peek["embeddings"][0]))
    except Exception as e:
        print("Error peeking ChromaDB:", e)
else:
    print("ChromaDB collection is None!")
