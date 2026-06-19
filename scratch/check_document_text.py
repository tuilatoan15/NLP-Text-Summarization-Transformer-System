import sqlite3
from pathlib import Path
import json
import sys

# Reconfigure stdout to use UTF-8
sys.stdout.reconfigure(encoding='utf-8')

db_path = Path(r"c:\Users\ASUS\Desktop\NLP-Text-Summarization-Transformer-System\storage\document_intelligence\rag\rag_chat.db")

def check_document_text():
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Let's search for the document first
    docs = cursor.execute("SELECT id, filename FROM rag_documents WHERE filename LIKE '%Hoiana_Aquaman%'").fetchall()
    print("=== DOCUMENTS MATCHING HOIANA AQUAMAN ===")
    for d in docs:
        print(f"ID: {d['id']} | Filename: {d['filename']}")
        
        # Let's get chunks for this document
        chunks = cursor.execute("SELECT id, chunk_index, text_content, metadata_json FROM rag_chunks WHERE document_id = ? ORDER BY chunk_index ASC", (d["id"],)).fetchall()
        print(f"Total Chunks: {len(chunks)}")
        for c in chunks:
            metadata = json.loads(c['metadata_json']) if c['metadata_json'] else {}
            chunk_type = metadata.get('chunk_type', 'base')
            print(f"\n--- Chunk Index: {c['chunk_index']} | Type: {chunk_type} | ID: {c['id']} ---")
            print(c['text_content'][:400] + "...")
            
    conn.close()

if __name__ == "__main__":
    check_document_text()
