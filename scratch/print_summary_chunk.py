import sqlite3
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')
db_path = Path(r"c:\Users\ASUS\Desktop\NLP-Text-Summarization-Transformer-System\storage\document_intelligence\rag\rag_chat.db")

def print_summary_chunk():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    row = cursor.execute("SELECT text_content FROM rag_chunks WHERE id = 'raptor_81c33bb3-ca15-46c1-b212-27b3b91d80a7_L1_0_bfbcc803'").fetchone()
    if row:
        print("=== EXACT TEXT CONTENT ===")
        print(repr(row[0]))
    conn.close()

if __name__ == "__main__":
    print_summary_chunk()
