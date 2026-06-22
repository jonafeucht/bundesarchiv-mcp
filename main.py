import asyncio
import json
import os
import tempfile
import threading
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
import fitz
import numpy as np
import uvicorn
from dotenv import load_dotenv
from mcp import types
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from sentence_transformers import SentenceTransformer
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Mount

from src.middleware.api_key import APIKeyMiddleware

load_dotenv()

PDF_DIR = Path(os.environ.get("PDF_DIR", "./pdfs")).resolve()
INDEX_PATH = Path(os.environ.get("INDEX_PATH", "./faiss_index"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 50))
TOP_K = int(os.environ.get("TOP_K", 8))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(
        f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be less than CHUNK_SIZE ({CHUNK_SIZE})"
    )


# ---------------------------------------------------------------------------
# VectorStore — encapsulates all index state and operations
# ---------------------------------------------------------------------------


class VectorStore:
    """Thread-safe wrapper around a FAISS index with JSON-serialised metadata."""

    def __init__(self, embed_model: str, index_path: Path) -> None:
        self._model = SentenceTransformer(embed_model)
        self._dim: int = self._model.get_sentence_embedding_dimension()
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(self._dim)
        self._chunks: list[dict] = []
        self._indexed_files: set[str] = set()
        self._mtimes: dict[str, float] = {}
        self._lock = threading.Lock()
        self._index_path = index_path

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Atomically persist the FAISS index and JSON metadata."""
        self._index_path.mkdir(parents=True, exist_ok=True)

        meta_target = self._index_path / "chunks.json"
        with tempfile.NamedTemporaryFile(
            "w",
            dir=self._index_path,
            delete=False,
            suffix=".tmp",
        ) as tmp:
            json.dump(
                {
                    "chunks": self._chunks,
                    "indexed_files": list(self._indexed_files),
                    "mtimes": self._mtimes,
                },
                tmp,
            )
            tmp_path = tmp.name
        os.replace(tmp_path, meta_target)

        idx_tmp = self._index_path / "index.faiss.tmp"
        faiss.write_index(self._index, str(idx_tmp))
        os.replace(idx_tmp, self._index_path / "index.faiss")

    def load(self) -> bool:
        """Load a previously saved index. Returns True on success."""
        idx_file = self._index_path / "index.faiss"
        meta_file = self._index_path / "chunks.json"

        if not idx_file.exists() or not meta_file.exists():
            return False

        self._index = faiss.read_index(str(idx_file))

        with open(meta_file) as f:
            data = json.load(f)

        self._chunks = data["chunks"]
        self._indexed_files = set(data["indexed_files"])
        self._mtimes = data.get("mtimes", {})
        print(
            f"Loaded existing index: {len(self._chunks)} chunks, "
            f"{len(self._indexed_files)} files."
        )
        return True

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _remove_file_chunks(self, filename: str) -> None:
        """Remove all chunks for a file and rebuild the FAISS index.

        Must be called with self._lock held.
        Rebuilding is O(n) but keeps the implementation simple; for very large
        indexes a more surgical approach (IDMap2 + remove_ids) could be used.
        """
        self._chunks = [c for c in self._chunks if c["filename"] != filename]
        self._indexed_files.discard(filename)
        self._mtimes.pop(filename, None)

        self._index = faiss.IndexFlatIP(self._dim)
        if self._chunks:
            texts = [c["text"] for c in self._chunks]
            embeddings = self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            self._index.add(np.array(embeddings, dtype="float32"))

    def index_pdf(self, pdf_path: Path) -> None:
        filename = str(pdf_path.relative_to(PDF_DIR))
        mtime = pdf_path.stat().st_mtime

        with self._lock:
            if filename in self._indexed_files and self._mtimes.get(filename) == mtime:
                return

        print(f"Indexing {filename}...")
        text = extract_text(pdf_path)
        chunks = chunk_text(text)

        if not chunks:
            print(
                f"Warning: No text extracted from {filename}. Skipping vector generation."
            )
            with self._lock:
                self._indexed_files.add(filename)
                self._mtimes[filename] = mtime
            return

        embeddings = self._model.encode(
            chunks, normalize_embeddings=True, show_progress_bar=False
        )
        embeddings = np.array(embeddings, dtype="float32")

        with self._lock:
            if filename in self._indexed_files and self._mtimes.get(filename) == mtime:
                return
            if filename in self._indexed_files:
                self._remove_file_chunks(filename)
            self._index.add(embeddings)
            for i, chunk in enumerate(chunks):
                self._chunks.append({"filename": filename, "chunk": i, "text": chunk})
            self._indexed_files.add(filename)
            self._mtimes[filename] = mtime

        print(f"  → {len(chunks)} chunks indexed")

    def index_all_pdfs(self) -> None:
        if not self.load():
            print("No existing index found, building from scratch...")

        pdfs = list(PDF_DIR.rglob("*.pdf"))
        new_or_changed = [
            p
            for p in pdfs
            if str(p.relative_to(PDF_DIR)) not in self._indexed_files
            or self._mtimes.get(str(p.relative_to(PDF_DIR))) != p.stat().st_mtime
        ]

        if not new_or_changed:
            print(f"Index up to date ({len(pdfs)} PDFs).")
            return

        print(f"Indexing {len(new_or_changed)} new/changed PDFs...")
        for pdf in new_or_changed:
            self.index_pdf(pdf)

        self.save()
        print("Indexing complete.")

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def list_pdfs(self) -> list[str]:
        return sorted(str(f.relative_to(PDF_DIR)) for f in PDF_DIR.rglob("*.pdf"))

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        filename_filter: str | None = None,
    ) -> list[tuple[float, dict]]:
        with self._lock:
            total = len(self._chunks)

        if total == 0:
            return []

        query_vec = self._model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype="float32")

        fetch_k = min(top_k * 10 if filename_filter else top_k, total)

        with self._lock:
            scores, indices = self._index.search(query_vec, fetch_k)
            chunks_snapshot = self._chunks

        results: list[tuple[float, dict]] = []
        seen: set[tuple[str, int]] = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = chunks_snapshot[idx]
            key = (meta["filename"], meta["chunk"])
            if key in seen:
                continue
            if filename_filter and meta["filename"] != filename_filter:
                continue
            seen.add(key)
            results.append((float(score), meta))
            if len(results) == top_k:
                break

        return results


# ---------------------------------------------------------------------------
# Pure helpers (no global state)
# ---------------------------------------------------------------------------


def extract_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    i = 0
    stride = size - overlap
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += stride
    return chunks


def resolve_pdf_path(filename: str) -> Path:
    """Resolve and validate a user-supplied filename against PDF_DIR.

    Raises ValueError on path traversal attempts.
    """
    resolved = (PDF_DIR / filename).resolve()
    if not resolved.is_relative_to(PDF_DIR):
        raise ValueError(f"Invalid filename: {filename!r}")
    return resolved


# ---------------------------------------------------------------------------
# Singleton store + MCP server
# ---------------------------------------------------------------------------

_store = VectorStore(EMBED_MODEL, INDEX_PATH)
server = Server("Akten der Reichskanzlei - Bundesarchiv MCP")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="list_pdfs",
            description="List all available PDFs.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="search",
            description=(
                "Semantic search across all PDFs. Returns the most relevant passages "
                "for a natural-language query. Optionally filter to a single file."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filename": {
                        "type": "string",
                        "description": "Optional: restrict search to this PDF.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": f"Number of results to return (default {TOP_K}).",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_pdf_text",
            description="Return the full extracted text of a single PDF.",
            inputSchema={
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    def _err(msg: str) -> list[types.TextContent]:
        return [types.TextContent(type="text", text=f"Error: {msg}")]

    try:
        if name == "list_pdfs":
            pdfs = _store.list_pdfs()
            text = "\n".join(pdfs) if pdfs else "(no PDFs found)"
            return [types.TextContent(type="text", text=text)]

        elif name == "search":
            query = arguments.get("query", "").strip()
            if not query:
                return _err("'query' must be a non-empty string.")

            top_k = int(arguments.get("top_k", TOP_K))
            filename_filter = arguments.get("filename")

            results = _store.search(query, top_k=top_k, filename_filter=filename_filter)

            if not results:
                return [types.TextContent(type="text", text="No results found.")]

            lines = [
                f"[{m['filename']} | chunk {m['chunk']} | score {s:.3f}]\n{m['text']}"
                for s, m in results
            ]
            return [types.TextContent(type="text", text="\n\n---\n\n".join(lines))]

        elif name == "get_pdf_text":
            filename = arguments.get("filename", "")
            try:
                pdf = resolve_pdf_path(filename)
            except ValueError as exc:
                return _err(str(exc))

            if not pdf.exists():
                return _err(f"PDF not found: {filename!r}")

            return [types.TextContent(type="text", text=extract_text(pdf))]

        return _err(f"Unknown tool: {name!r}")

    except Exception as exc:
        return _err(f"Unexpected error in tool '{name}': {exc}")


# ---------------------------------------------------------------------------
# Starlette app
# ---------------------------------------------------------------------------

session_manager = StreamableHTTPSessionManager(
    app=server,
    event_store=None,
    json_response=True,
    stateless=True,
)


@asynccontextmanager
async def lifespan(_app):
    await asyncio.get_event_loop().run_in_executor(None, _store.index_all_pdfs)
    async with session_manager.run():
        yield


async def handle_mcp(scope, receive, send):
    await session_manager.handle_request(scope, receive, send)


app = Starlette(
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        ),
        Middleware(APIKeyMiddleware),
    ],
    routes=[Mount("/", app=handle_mcp)],
    lifespan=lifespan,
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
