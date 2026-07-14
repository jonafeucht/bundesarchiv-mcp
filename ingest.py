import hashlib
import os
from pathlib import Path

import fitz
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(os.getenv("PDF_DIR", "./mcp-data")).resolve()
OUTPUT_DIR = Path(os.getenv("DB_URI", "./mcp-server/lancedb_index")).resolve()

TABLE_NAME = "document_chunks"
EMBED_MODEL = "jinaai/jina-embeddings-v5-text-nano"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    stride = size - overlap
    chunks = []

    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += stride

    return chunks


def extract_text_from_pdf(path: Path) -> str:
    with fitz.open(path) as doc:
        return "".join(page.get_text() for page in doc)


def main():
    print("Scanning files...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(
        EMBED_MODEL, trust_remote_code=True, model_kwargs={"default_task": "retrieval"}
    )

    db = lancedb.connect(OUTPUT_DIR)

    if TABLE_NAME in db.table_names():
        table = db.open_table(TABLE_NAME)
        existing = table.to_pandas()

        existing_hashes = (
            set(existing["file_hash"].unique())
            if "file_hash" in existing.columns
            else set()
        )
    else:
        table = None
        existing_hashes = set()

    new_chunks = []

    for file_path in DATA_DIR.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in [".pdf", ".txt", ".md"]:
            continue

        relative_name = str(file_path.relative_to(DATA_DIR))
        file_hash = hash_file(file_path)

        if file_hash in existing_hashes:
            print(f"Skipping unchanged: {relative_name}")
            continue

        print(f"Processing: {relative_name}")

        if file_path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(file_path)
        else:
            text = file_path.read_text(encoding="utf-8")

        for i, chunk in enumerate(chunk_text(text)):
            new_chunks.append(
                {
                    "filename": relative_name,
                    "file_hash": file_hash,
                    "chunk": i,
                    "text": chunk,
                }
            )

    if not new_chunks:
        print("No new or changed files found.")
        return

    print(f"Embedding {len(new_chunks)} chunks in increments...")

    db_batch_size = 1000

    for idx in range(0, len(new_chunks), db_batch_size):
        batch = new_chunks[idx : idx + db_batch_size]
        batch_texts = [c["text"] for c in batch]

        embeddings = model.encode(
            batch_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=16,
            show_progress_bar=False,
        ).astype("float32")

        for j, emb in enumerate(embeddings):
            batch[j]["vector"] = emb.tolist()

        df = pd.DataFrame(batch)

        if table is None:
            print("Creating table with initial batch...")
            table = db.create_table(TABLE_NAME, data=df)
        else:
            try:
                table.add(df)
            except ValueError as e:
                if "Cast error" in str(e) or "Cannot cast" in str(e):
                    print("\nSchema mismatch detected. Re-creating table...")
                    db.drop_table(TABLE_NAME)
                    table = db.create_table(TABLE_NAME, data=df)
                else:
                    raise e

        print(f"Indexed chunks {idx} to {idx + len(batch)}...")

    print(f"✔ Done. Successfully processed all {len(new_chunks)} chunks.")


if __name__ == "__main__":
    main()
