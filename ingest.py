import hashlib
import json
from pathlib import Path

import fitz
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("./mcp-data").resolve()
OUTPUT_DIR = Path("./mcp-server/lancedb_index").resolve()
CACHE_FILE = OUTPUT_DIR / "hashes_cache.json"
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


def calculate_sha256(path):
    """Calculates SHA-256 hash of a file to reliably detect content changes."""
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def main():
    print("Starting incremental ingestion process into LanceDB...")
    model = SentenceTransformer(EMBED_MODEL)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_hashes = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                existing_hashes = json.load(f)
        except Exception:
            existing_hashes = {}

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

        file_hash = calculate_sha256(file_path)
        current_files[relative_name] = file_hash

        if (
            relative_name in existing_hashes
            and existing_hashes[relative_name] == file_hash
        ):
            continue

        files_to_process.append((file_path, relative_name, file_hash))

    deleted_files = [f for f in existing_hashes if f not in current_files]

    db = lancedb.connect(OUTPUT_DIR)
    table_exists = TABLE_NAME in db.list_tables()

    if not files_to_process and not deleted_files and table_exists:
        print("LanceDB index is completely up to date. No operations required.")
        return

    table = db.open_table(TABLE_NAME) if table_exists else None

    if deleted_files and table_exists:
        print(f"Purging {len(deleted_files)} deleted files from index...")
        for filename in deleted_files:
            table.delete_where(f"filename = '{filename}'")

    new_chunks = []
    updated_hashes = {
        f: existing_hashes[f] for f in existing_hashes if f in current_files
    }

    if files_to_process:
        print(f"Processing {len(files_to_process)} modified/new files...")
        for file_path, relative_name, file_hash in files_to_process:
            if table_exists and relative_name in existing_hashes:
                table.delete_where(f"filename = '{relative_name}'")

            if file_path.suffix.lower() == ".pdf":
                text = extract_text_from_pdf(file_path)
            else:
                text = file_path.read_text(encoding="utf-8")

            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                new_chunks.append(
                    {"filename": relative_name, "chunk": i, "text": chunk}
                )

            updated_hashes[relative_name] = file_hash

    if new_chunks:
        print(f"Computing embeddings for {len(new_chunks)} new chunks...")
        texts = [c["text"] for c in new_chunks]
        embeddings = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True
        )

        for idx, emb in enumerate(embeddings):
            new_chunks[idx]["vector"] = emb.tolist()

        df = pd.DataFrame(new_chunks)

        if table_exists:
            table.add(df)
        else:
            db.create_table(TABLE_NAME, data=df)
    elif not table_exists or len(current_files) == 0:
        print("No documents found anywhere. Cleaning storage database.")
        if table_exists:
            db.drop_table(TABLE_NAME)
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        return

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(updated_hashes, f, indent=2)

    print("LanceDB incremental database update finalized successfully!")


if __name__ == "__main__":
    main()
