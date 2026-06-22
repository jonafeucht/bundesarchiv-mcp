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
    print("Starting incremental ingestion process...")
    model = SentenceTransformer(EMBED_MODEL)

    meta_file = OUTPUT_DIR / "chunks.json"
    idx_file = OUTPUT_DIR / "index.faiss"

    existing_chunks = []
    existing_mtimes = {}

    if meta_file.exists() and idx_file.exists():
        try:
            print("Found existing index files. Loading cache...")
            with open(meta_file, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            existing_chunks = old_data.get("chunks", [])
            existing_mtimes = old_data.get("mtimes", {})
            print(f"Loaded cache containing {len(existing_mtimes)} tracking states.")
        except Exception as e:
            print(
                f"Warning: Failed to parse cache metadata ({e}). Rebuilding full index."
            )
            existing_chunks = []
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

    purged_chunks = [
        c
        for c in existing_chunks
        if c["filename"] in current_files
        and c["filename"] not in [f[1] for f in files_to_process]
    ]

    new_chunks = []
    updated_mtimes = {
        c["filename"]: existing_mtimes[c["filename"]]
        for c in purged_chunks
        if c["filename"] in existing_mtimes
    }

    if files_to_process:
        print(f"Found {len(files_to_process)} new or modified files to process.")
        for file_path, relative_name, mtime in files_to_process:
            text = ""
            if file_path.suffix.lower() == ".pdf":
                print(f"Processing PDF: {relative_name}")
                text = extract_text_from_pdf(file_path)
            elif file_path.suffix.lower() in [".txt", ".md"]:
                print(f"Processing Text/Markdown: {relative_name}")
                text = file_path.read_text(encoding="utf-8")

            chunks = chunk_text(text)
            if chunks:
                updated_mtimes[relative_name] = mtime
                for i, chunk in enumerate(chunks):
                    new_chunks.append(
                        {"filename": relative_name, "chunk": i, "text": chunk}
                    )
    else:
        print(
            "All existing files match the tracking cache. Checking index structure completeness..."
        )

    if (
        not new_chunks
        and len(purged_chunks) == len(existing_chunks)
        and idx_file.exists()
    ):
        print("Index is 100% up to date. No operations required.")
        return

    print("Reassembling text mapping matrices...")
    final_chunks = purged_chunks + new_chunks
    indexed_files = list(updated_mtimes.keys())

    if not final_chunks:
        print("No structural text items remain across files. Clearing outputs.")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if meta_file.exists():
            meta_file.unlink()
        if idx_file.exists():
            idx_file.unlink()
        return

    print(
        f"Building updated FAISS index context mapping ({len(final_chunks)} total chunks)..."
    )
    texts = [c["text"] for c in final_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(idx_file))

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "chunks": final_chunks,
                "indexed_files": indexed_files,
                "mtimes": updated_mtimes,
            },
            f,
            indent=2,
        )

    print(f"Incremental update finalized! Indexed: {len(indexed_files)} files total.")


if __name__ == "__main__":
    main()
