import hashlib
import os
from pathlib import Path

import fitz
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_DIR = Path(os.getenv("PDF_DIR", "./mcp-data")).resolve()
OUTPUT_DIR = Path(os.getenv("DB_URI", "./mcp-server/lancedb_index")).resolve()

TABLE_NAME = "document_chunks"
EMBED_MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBED_BATCH_SIZE = 16
DB_FLUSH_ROWS = 500
HASH_READ_CHUNK = 1024 * 1024


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(HASH_READ_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if not words:
        return []
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


def flush_to_db(db, table, rows):
    if not rows:
        return table

    texts = [r["text"] for r in rows]
    embeddings = MODEL.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
    ).astype("float32")

    for row, emb in zip(rows, embeddings):
        row["vector"] = emb.tolist()

    df = pd.DataFrame(rows)

    if table is None:
        table = db.create_table(TABLE_NAME, data=df)
    else:
        try:
            table.add(df)
        except ValueError as e:
            if "Cast error" in str(e) or "Cannot cast" in str(e):
                print("\n⚠️ Schema mismatch. Dropping and re-creating table...")
                db.drop_table(TABLE_NAME)
                table = db.create_table(TABLE_NAME, data=df)
            else:
                raise

    return table


def main():
    global MODEL

    print("🔍 Scanning files in", DATA_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("📥 Loading embedding model with ONNX optimization...")
    MODEL = SentenceTransformer(EMBED_MODEL, backend="onnx")

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

    print("📋 Discovering files...")
    candidate_paths = [
        p
        for p in DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in [".pdf", ".txt", ".md"]
    ]

    files_to_process = []
    for file_path in tqdm(candidate_paths, desc="Hashing files", unit="file"):
        file_hash = hash_file(file_path)
        if file_hash not in existing_hashes:
            files_to_process.append((file_path, file_hash))

    if not files_to_process:
        print("✅ No new or changed files found.")
        return

    print(f"📄 Found {len(files_to_process)} new/changed file(s) to process.")

    pending_rows = []
    processed = 0
    skipped = 0

    for file_path, file_hash in tqdm(
        files_to_process, desc="Processing Files", unit="file"
    ):
        relative_name = str(file_path.relative_to(DATA_DIR))

        try:
            if file_path.suffix.lower() == ".pdf":
                text = extract_text_from_pdf(file_path)
            else:
                text = file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"⚠️ Error reading {relative_name}: {e}. Skipping.")
            skipped += 1
            continue

        chunks = chunk_text(text)
        if not chunks:
            continue

        for i, chunk in enumerate(chunks):
            pending_rows.append(
                {
                    "filename": relative_name,
                    "file_hash": file_hash,
                    "chunk": i,
                    "text": chunk,
                }
            )

        processed += 1

        if len(pending_rows) >= DB_FLUSH_ROWS:
            table = flush_to_db(db, table, pending_rows)
            pending_rows = []

    table = flush_to_db(db, table, pending_rows)

    print(
        f"🎉 Done! Index updated successfully. Processed {processed} file(s), skipped {skipped}."
    )


if __name__ == "__main__":
    main()
