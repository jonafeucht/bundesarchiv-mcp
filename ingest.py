import json
from pathlib import Path

import fitz
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("./mcp-data").resolve()
OUTPUT_DIR = Path("./mcp-server/lancedb_index").resolve()
CACHE_FILE = OUTPUT_DIR / "mtimes_cache.json"
TABLE_NAME = "document_chunks"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    stride = size - overlap
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += stride
    return chunks


def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


def main():
    print("Starting ingestion process into LanceDB...")
    model = SentenceTransformer(EMBED_MODEL)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_mtimes = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                existing_mtimes = json.load(f)
        except Exception:
            existing_mtimes = {}

    if not DATA_DIR.exists():
        print(f"Error: Data directory {DATA_DIR} does not exist.")
        return

    current_files = {}
    files_to_process = []

    for file_path in DATA_DIR.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in [".pdf", ".txt", ".md"]:
            continue

        relative_name = str(file_path.relative_to(DATA_DIR))
        mtime = file_path.stat().st_mtime
        current_files[relative_name] = mtime

        if relative_name in existing_mtimes and existing_mtimes[relative_name] == mtime:
            continue

        files_to_process.append((file_path, relative_name, mtime))

    db = lancedb.connect(OUTPUT_DIR)
    table_exists = TABLE_NAME in db.table_names()

    if not files_to_process and table_exists:
        if set(existing_mtimes.keys()) == set(current_files.keys()):
            print("LanceDB index is up to date. No operations required.")
            return

    print("Processing structural items...")

    all_chunks = []
    updated_mtimes = {}

    for file_path in DATA_DIR.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in [
            ".pdf",
            ".txt",
            ".md",
        ]:
            continue

        relative_name = str(file_path.relative_to(DATA_DIR))
        mtime = file_path.stat().st_mtime
        updated_mtimes[relative_name] = mtime

        if file_path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(file_path)
        else:
            text = file_path.read_text(encoding="utf-8")

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({"filename": relative_name, "chunk": i, "text": chunk})

    if not all_chunks:
        print("No documents found. Cleaning storage database.")
        if table_exists:
            db.drop_table(TABLE_NAME)
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        return

    print(f"Computing embeddings for {len(all_chunks)} chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    for idx, emb in enumerate(embeddings):
        all_chunks[idx]["vector"] = emb.tolist()

    df = pd.DataFrame(all_chunks)

    db.create_table(TABLE_NAME, data=df, mode="overwrite")

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(updated_mtimes, f, indent=2)

    print("LanceDB database update finalized successfully!")


if __name__ == "__main__":
    main()
