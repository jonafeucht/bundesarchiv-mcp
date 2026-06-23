import os
from pathlib import Path

import fitz
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(os.getenv("PDF_DIR", "./mcp-data")).resolve()
OUTPUT_DIR = Path(os.getenv("DB_URI", "./mcp-server/lancedb_index")).resolve()

TABLE_NAME = "document_chunks"
EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
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

    chunks = []
    for file_path in DATA_DIR.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in [".pdf", ".txt", ".md"]:
            continue

        relative_name = str(file_path.relative_to(DATA_DIR))
        print(f"Processing: {relative_name}")

        if file_path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(file_path)
        else:
            text = file_path.read_text(encoding="utf-8")

        for i, chunk in enumerate(chunk_text(text)):
            chunks.append({"filename": relative_name, "chunk": i, "text": chunk})

    if not chunks:
        print("ERROR: No files found in", DATA_DIR)
        raise SystemExit(1)

    print(f"Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    for i, emb in enumerate(embeddings):
        chunks[i]["vector"] = emb.tolist()

    db = lancedb.connect(OUTPUT_DIR)

    if TABLE_NAME in db.list_tables():
        db.drop_table(TABLE_NAME)

    db.create_table(TABLE_NAME, data=pd.DataFrame(chunks))
    print(f"✔ Done. {len(chunks)} chunks written to '{TABLE_NAME}'")


if __name__ == "__main__":
    main()
