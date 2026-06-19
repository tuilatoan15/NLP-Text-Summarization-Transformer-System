import chromadb
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')
project_root = Path(r"c:\Users\ASUS\Desktop\NLP-Text-Summarization-Transformer-System")

def check_chromadb_chunk():
    chroma_client = chromadb.PersistentClient(path=str(project_root / "storage" / "document_intelligence" / "rag" / "chroma"))
    collection = chroma_client.get_collection("rag_chunks")
    
    res = collection.get(ids=["raptor_81c33bb3-ca15-46c1-b212-27b3b91d80a7_L1_0_bfbcc803"], include=["documents", "metadatas"])
    print("=== CHROMADB CHUNK ===")
    print("IDs:", res["ids"])
    print("Documents count:", len(res["documents"]))
    if res["documents"]:
        print("Document Text Content:")
        print(repr(res["documents"][0]))
        
if __name__ == "__main__":
    check_chromadb_chunk()
