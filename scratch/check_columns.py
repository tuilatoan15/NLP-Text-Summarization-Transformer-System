import sqlite3
from pathlib import Path

db_path = Path(r"c:\Users\ASUS\Desktop\NLP-Text-Summarization-Transformer-System\storage\document_intelligence\rag\rag_chat.db")

def check_columns():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(rag_chunks)")
    print("rag_chunks columns:")
    for col in cursor.fetchall():
        print(col)
    conn.close()

if __name__ == "__main__":
    check_columns()
