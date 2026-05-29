import sys
try:
    import chromadb
    print("chromadb successfully imported!")
    print("chromadb version:", chromadb.__version__)
except Exception as e:
    import traceback
    print("Failed to import chromadb:")
    traceback.print_exc()
