import gc
import hashlib
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, TimeoutError as ConnTimeoutError
from pathlib import Path

import fitz
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = Path(os.getenv("PDF_DIR", "./mcp-data")).resolve()
OUTPUT_DIR = Path(os.getenv("DB_URI", "./mcp-server/lancedb_index")).resolve()

TABLE_NAME = "document_chunks"
EMBED_MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBED_BATCH_SIZE = 16
DB_FLUSH_ROWS = 500
HASH_READ_CHUNK = 1024 * 1024
FILE_TIMEOUT_SECONDS = 20
MAX_TEXT_SIZE_BYTES = 50 * 1024 * 1024


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for block in iter(lambda: f.read(HASH_READ_CHUNK), b""):
                h.update(block)
        return h.hexdigest()
    except Exception as e:
        print(f"\n⚠️ Error hashing {path.name}: {e}")
        return ""


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


def extract_and_chunk_single_file(file_info):
    file_path, file_hash, relative_name = file_info
    try:
        if file_path.suffix.lower() == ".pdf":
            text_chunks = []
            with fitz.open(file_path) as doc:
                for page in doc:
                    text_chunks.append(page.get_text())
            text = "".join(text_chunks)
        else:
            if file_path.stat().st_size > MAX_TEXT_SIZE_BYTES:
                raise ValueError(
                    f"File exceeds limit: {MAX_TEXT_SIZE_BYTES / 1024 / 1024} MB"
                )
            text = file_path.read_text(encoding="utf-8", errors="ignore")

        chunks = chunk_text(text)
        payloads = [
            {
                "filename": relative_name,
                "file_hash": file_hash,
                "chunk": i,
                "text": chunk,
            }
            for i, chunk in enumerate(chunks)
        ]

        return "SUCCESS", relative_name, payloads
    except Exception as e:
        return "ERROR", relative_name, str(e)


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
        if file_hash and file_hash not in existing_hashes:
            files_to_process.append((file_path, file_hash))

    if not files_to_process:
        print("✅ No new or changed files found.")
        return

    print(f"📄 Found {len(files_to_process)} new/changed file(s) to process.")

    tasks = []
    for file_path, file_hash in files_to_process:
        relative_name = str(file_path.relative_to(DATA_DIR))
        tasks.append((file_path, file_hash, relative_name))

    pending_rows = []
    processed = 0
    skipped = 0

    max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"⚡ Starting parallel parser using {max_workers} worker processes...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_filename = {
            executor.submit(extract_and_chunk_single_file, t): t[2] for t in tasks
        }

        progress_bar = tqdm(
            total=len(future_to_filename), desc="Processing Files", unit="file"
        )

        for future in list(future_to_filename.keys()):
            file_name = future_to_filename[future]
            try:
                status, _, result = future.result(timeout=FILE_TIMEOUT_SECONDS)

                if status == "SUCCESS":
                    progress_bar.write(f"⚙️ Processed: {file_name}")
                    pending_rows.extend(result)
                    processed += 1
                else:
                    progress_bar.write(f"⚠️ Skipped: {file_name} | Reason: {result}")
                    skipped += 1
            except ConnTimeoutError:
                progress_bar.write(
                    f"🚨 Timeout: {file_name} took longer than {FILE_TIMEOUT_SECONDS}s and was skipped."
                )
                future.cancel()
                skipped += 1
            except Exception as e:
                progress_bar.write(f"💥 Failed: {file_name} | Error: {e}")
                skipped += 1
            finally:
                progress_bar.update(1)

            if len(pending_rows) >= DB_FLUSH_ROWS:
                to_flush = pending_rows[:DB_FLUSH_ROWS]
                table = flush_to_db(db, table, to_flush)
                pending_rows = pending_rows[DB_FLUSH_ROWS:]

        if pending_rows:
            table = flush_to_db(db, table, pending_rows)

    print(
        f"🎉 Done! Index updated successfully. Processed {processed} file(s), skipped {skipped}."
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
