# ingest.py
from pathlib import Path

from main import EMBED_MODEL, INDEX_PATH, VectorStore

if __name__ == "__main__":
    print("Pre-building FAISS index for deployment...")
    store = VectorStore(EMBED_MODEL, INDEX_PATH)
    # This will load existing or build fresh from your local/CI directory
    store.index_all_pdfs()
    print("FAISS index successfully built and saved!")
