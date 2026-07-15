import gc
import hashlib
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

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
FILE_TIMEOUT_SECONDS = 20
MAX_TEXT_SIZE_BYTES = 10 * 1024 * 1024
MAX_PDF_SIZE_BYTES = 25 * 1024 * 1024


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for block in iter(lambda: f.read(HASH_READ_CHUNK), b""):
                h.update(block)
        return h.hexdigest()
    except Exception as e:
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


def extract_hash_and_chunk(file_path: Path, relative_name: str):
    try:
        if file_path.suffix.lower() == ".pdf":
            if file_path.stat().st_size > MAX_PDF_SIZE_BYTES:
                return "SKIP", relative_name, "File size too large"

            with fitz.open(file_path) as doc:
                if doc.page_count > 500:
                    return "SKIP", relative_name, "Too many pages"
                text_chunks = []
                for page in doc:
                    text_chunks.append(page.get_text())
            text = "".join(text_chunks)
        else:
            if file_path.stat().st_size > MAX_TEXT_SIZE_BYTES:
                return "SKIP", relative_name, "File size too large"
            text = file_path.read_text(encoding="utf-8", errors="ignore")

        file_hash = hash_file(file_path)
        if not file_hash:
            return "SKIP", relative_name, "Hashing failed"

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

        return "SUCCESS", relative_name, (file_hash, payloads)
    except Exception as e:
        return "ERROR", relative_name, str(e)


def flush_to_db(db, table, rows, model):
    if not rows:
        return table

    texts = [r["text"] for r in rows]
    embeddings = model.encode(
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
                db.drop_table(TABLE_NAME)
                table = db.create_table(TABLE_NAME, data=df)
            else:
                raise

    return table


def main():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    print("🔍 Scanning files in", DATA_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    db = lancedb.connect(OUTPUT_DIR)
    existing_hashes = set()

    if TABLE_NAME in db.table_names():
        table = db.open_table(TABLE_NAME)
        existing_hashes = set(
            table.search().select(["file_hash"]).to_pandas()["file_hash"].unique()
        )
    else:
        table = None

    candidate_paths = []
    for p in DATA_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".pdf", ".txt", ".md"]:
            if p.suffix.lower() == ".pdf" and p.stat().st_size > MAX_PDF_SIZE_BYTES:
                continue
            if p.suffix.lower() != ".pdf" and p.stat().st_size > MAX_TEXT_SIZE_BYTES:
                continue
            candidate_paths.append(p)

    print(
        f"📋 Found {len(candidate_paths)} clean candidate files. Initializing workers..."
    )

    print("📥 Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL, backend="onnx")

    pending_rows = []
    processed = 0
    skipped = 0

    max_workers = max(1, multiprocessing.cpu_count() - 2)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {}
        for p in candidate_paths:
            rel_name = str(p.relative_to(DATA_DIR))
            future = executor.submit(extract_hash_and_chunk, p, rel_name)
            future_to_file[future] = rel_name

        progress_bar = tqdm(
            total=len(future_to_file), desc="Processing Files", unit="file"
        )

        for future in as_completed(future_to_file.keys()):
            file_name = future_to_file[future]
            try:
                status, _, result = future.result(timeout=FILE_TIMEOUT_SECONDS)

                if status == "SUCCESS":
                    file_hash, payloads = result
                    if file_hash in existing_hashes:
                        skipped += 1
                    else:
                        pending_rows.extend(payloads)
                        processed += 1
                else:
                    skipped += 1

            except Exception as e:
                skipped += 1
            finally:
                progress_bar.update(1)

            if len(pending_rows) >= DB_FLUSH_ROWS:
                to_flush = pending_rows[:DB_FLUSH_ROWS]
                table = flush_to_db(db, table, to_flush, model)
                pending_rows = pending_rows[DB_FLUSH_ROWS:]

        if pending_rows:
            table = flush_to_db(db, table, pending_rows, model)

    print(f"🎉 Done! Processed {processed} file(s), skipped/unchanged {skipped}.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
