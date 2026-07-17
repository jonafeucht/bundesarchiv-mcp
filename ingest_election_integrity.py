import hashlib
import os
from pathlib import Path

import fitz
import lancedb
import pandas as pd
import pyarrow as pa
from sentence_transformers import SentenceTransformer

# Updated defaults for Election Integrity
DATA_DIR = Path(os.getenv("PDF_DIR", "./election-integrity")).resolve()
OUTPUT_DIR = Path(
    os.getenv("DB_URI", "./mcp-server/lancedb_index_election_integrity")
).resolve()

TABLE_NAME = "document_chunks"
EMBED_MODEL = "intfloat/multilingual-e5-small"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
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

    model = SentenceTransformer(EMBED_MODEL)
    db = lancedb.connect(OUTPUT_DIR)

    existing_hashes = set()
    table = None

    if TABLE_NAME in db.table_names():
        table = db.open_table(TABLE_NAME)
        existing_hashes = set(
            table.search()
            .select(["file_hash"])
            .limit(10_000_000)
            .to_arrow()
            .column("file_hash")
            .to_pylist()
        )

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

    print(f"Embedding {len(new_chunks)} chunks...")
    texts = [c["text"] for c in new_chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=False,
        batch_size=32,
        show_progress_bar=True,
    )

    embeddings_list = [emb.tolist() for emb in embeddings]

    for i, emb in enumerate(embeddings_list):
        new_chunks[i]["vector"] = emb

    df = pd.DataFrame(new_chunks)
    arrow_table = pa.Table.from_pandas(df)

    if table is None:
        print("Creating table...")
        db.create_table(TABLE_NAME, data=arrow_table)
    else:
        print("Appending to existing table...")
        table.add(arrow_table)

    print(f"✔ Done. Added {len(new_chunks)} new chunks.")


if __name__ == "__main__":
    main()
