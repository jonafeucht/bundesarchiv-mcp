import json
import os
from pathlib import Path

import faiss
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("./mcp-data").resolve()
OUTPUT_DIR = Path("./mcp-server/faiss_index").resolve()
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
    print("Starting ingestion process...")
    model = SentenceTransformer(EMBED_MODEL)
    all_chunks = []
    indexed_files = set()

    if not DATA_DIR.exists():
        print(f"Error: Data directory {DATA_DIR} does not exist.")
        return

    for file_path in DATA_DIR.rglob("*"):
        if not file_path.is_file():
            continue

        relative_name = str(file_path.relative_to(DATA_DIR))

        text = ""
        if file_path.suffix.lower() == ".pdf":
            print(f"Processing PDF: {relative_name}")
            text = extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() in [".txt", ".md"]:
            print(f"Processing Text/Markdown: {relative_name}")
            text = file_path.read_text(encoding="utf-8")
        else:
            continue

        chunks = chunk_text(text)
        if chunks:
            indexed_files.add(relative_name)
            for i, chunk in enumerate(chunks):
                all_chunks.append(
                    {"filename": relative_name, "chunk": i, "text": chunk}
                )

    if not all_chunks:
        print("No text data found to index.")
        return

    print(f"Generating embeddings for {len(all_chunks)} text chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    print("Building FAISS index matrix...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(OUTPUT_DIR / "index.faiss"))

    with open(OUTPUT_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(
            {"chunks": all_chunks, "indexed_files": list(indexed_files)}, f, indent=2
        )

    print(f"Ingestion complete! Successfully indexed {len(indexed_files)} files.")


if __name__ == "__main__":
    main()
