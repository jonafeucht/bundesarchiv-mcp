import hashlib
import json
import os
from pathlib import Path

import fitz
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(os.getenv("PDF_DIR", "./mcp-data")).resolve()
OUTPUT_DIR = Path(os.getenv("DB_URI", "./mcp-server/lancedb_index")).resolve()

CACHE_FILE = OUTPUT_DIR / "hashes_cache.json"
TABLE_NAME = "document_chunks"
EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def calculate_sha256(path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(block)
    return sha256_hash.hexdigest()


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    stride = size - overlap
    chunks = []

    i = 0
    while i < len(words):
        chunk = words[i : i + size]
        chunks.append(" ".join(chunk))
        i += stride

    return chunks


def extract_text_from_pdf(path: Path) -> str:
    with fitz.open(path) as doc:
        return "".join(page.get_text() for page in doc)


def load_cache():
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def scan_files(existing_hashes):
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

    return current_files, files_to_process, deleted_files


def table_exists(db):
    return TABLE_NAME in db.list_tables()


def main():
    print("Scanning files for changes...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_hashes = load_cache()
    current_files, files_to_process, deleted_files = scan_files(existing_hashes)

    db = lancedb.connect(OUTPUT_DIR)
    has_table = table_exists(db)

    if not files_to_process and not deleted_files and has_table:
        print("✔ No changes detected. Skipping embedding + DB work.")
        return

    print("Changes detected:")
    print(f"  - New/modified files: {len(files_to_process)}")
    print(f"  - Deleted files: {len(deleted_files)}")

    model = SentenceTransformer(EMBED_MODEL)

    table = db.open_table(TABLE_NAME) if has_table else None

    def delete_file_chunks(filename: str):
        safe = filename.replace("'", "''")
        table.delete_where(f"filename = '{safe}'")

    if has_table:
        for filename in deleted_files:
            print(f"Deleting removed file: {filename}")
            delete_file_chunks(filename)

        for _, relative_name, _ in files_to_process:
            print(f"Refreshing file: {relative_name}")
            delete_file_chunks(relative_name)

    new_chunks = []
    updated_hashes = {
        f: existing_hashes[f] for f in existing_hashes if f in current_files
    }

    for file_path, relative_name, file_hash in files_to_process:
        if file_path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(file_path)
        else:
            text = file_path.read_text(encoding="utf-8")

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            new_chunks.append({"filename": relative_name, "chunk": i, "text": chunk})

        updated_hashes[relative_name] = file_hash

    if new_chunks:
        print(f"Embedding {len(new_chunks)} chunks...")

        texts = [c["text"] for c in new_chunks]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")

        for i, emb in enumerate(embeddings):
            new_chunks[i]["vector"] = emb.tolist()

        df = pd.DataFrame(new_chunks)

        if has_table:
            table.add(df)
        else:
            db.create_table(TABLE_NAME, data=df)

    CACHE_FILE.write_text(json.dumps(updated_hashes, indent=2))

    print("✔ LanceDB incremental update complete")


if __name__ == "__main__":
    main()
