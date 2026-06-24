import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import lancedb
import uvicorn
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / ".env"

load_dotenv(dotenv_path=env_path, override=True)

from mcp import types
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from sentence_transformers import SentenceTransformer
from src.middleware.api_key import APIKeyMiddleware
from src.service.token_budget_service import apply_token_budget
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Mount

DB_URI = os.environ.get("DB_URI", "./lancedb_index")
TABLE_NAME = "document_chunks"

TOP_K = int(os.environ.get("TOP_K", "5"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

MAX_CHARS_PER_CHUNK = int(os.getenv("MAX_CHARS_PER_CHUNK", "6000"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "24000"))


class VectorStore:
    def __init__(self, embed_model: str, db_uri: str) -> None:
        self._model = SentenceTransformer(embed_model)
        self._db_uri = db_uri
        self._db = None
        self._table = None
        self._cached_filenames: list[str] = []

    def load(self) -> bool:
        try:
            self._db = lancedb.connect(self._db_uri)

            if TABLE_NAME not in self._db.table_names():
                print(f"Error: Table '{TABLE_NAME}' missing at {self._db_uri}")
                return False

            self._table = self._db.open_table(TABLE_NAME)

            print(f"Ready: LanceDB connected. Table size={len(self._table)}")
            self._refresh_filename_cache()
            return True

        except Exception as e:
            print(f"Failed to load LanceDB: {e}")
            return False

    def _refresh_filename_cache(self):
        """Pre-aggregates distinct files so list_pdfs pagination is instant."""
        try:
            df = (
                self._table.search()
                .where("filename IS NOT NULL")
                .select(["filename"])
                .to_pandas()
            )
            self._cached_filenames = sorted(df["filename"].dropna().unique().tolist())
        except Exception as e:
            print(f"Failed to refresh cache: {e}")
            self._cached_filenames = []

    def list_pdfs(self, page: int = 1, per_page: int = 50) -> list[str]:
        if not self._cached_filenames and self._table is not None:
            self._refresh_filename_cache()

        page = max(page, 1)
        per_page = max(min(per_page, 100), 1)
        offset = (page - 1) * per_page
        return self._cached_filenames[offset : offset + per_page]

    def total_files_count(self) -> int:
        return len(self._cached_filenames)

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        offset: int = 0,
        filename_filter: str | None = None,
    ) -> list[tuple[float, dict]]:
        if self._table is None:
            return []

        query_vec = self._model.encode(query, normalize_embeddings=True).tolist()

        qb = self._table.search(query_vec).limit(offset + top_k)

        if filename_filter:
            safe_filter = filename_filter.replace("'", "''")
            qb = qb.where(f"filename = '{safe_filter}'")

        df = qb.to_pandas()
        results: list[tuple[float, dict]] = []

        for _, row in df.iterrows():
            score = float(row.get("_distance", row.get("_score", 0.0)))
            results.append(
                (
                    score,
                    {
                        "filename": row["filename"],
                        "chunk": int(row["chunk"]),
                        "text": row["text"],
                    },
                )
            )

        return results[offset : offset + top_k]


_store = VectorStore(EMBED_MODEL, DB_URI)
server = Server("Akten für das Bundesarchiv")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="list_pdfs",
            description="List available indexed files with pagination.",
            inputSchema={
                "type": "object",
                "properties": {
                    "page": {
                        "type": "integer",
                        "description": "Page number (defaults to 1)",
                        "default": 1,
                    },
                    "per_page": {
                        "type": "integer",
                        "description": "Number of files per page (defaults to 50, max 100)",
                        "default": 50,
                    },
                },
            },
        ),
        types.Tool(
            name="search",
            description="Semantic search across all contexts. Supports paginating deeper chunks via offset.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filename": {
                        "type": "string",
                        "description": "Optional filter by filename",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": f"Number of records to return on this page (max {TOP_K}).",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Number of records to skip. Use multiples of top_k (e.g., 5, 10) to view next pages.",
                        "default": 0,
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    def err(msg: str):
        return [types.TextContent(type="text", text=f"Error: {msg}")]

    try:
        if name == "list_pdfs":
            page = max(int(arguments.get("page", 1)), 1)
            per_page = min(max(int(arguments.get("per_page", 50)), 1), 100)

            pdfs = _store.list_pdfs(page=page, per_page=per_page)
            total_count = _store.total_files_count()
            total_pages = (total_count + per_page - 1) // per_page

            if pdfs:
                text = (
                    f"--- Page {page} of {total_pages} (Total files: {total_count}) ---\n"
                    + "\n".join(pdfs)
                )
                if page < total_pages:
                    text += f"\n\n[SYSTEM NOTE]: More files exist. To see the next files, call 'list_pdfs' with page={page + 1}."
            else:
                text = f"(No files found on page {page})"

            return [types.TextContent(type="text", text=text)]

        elif name == "search":
            query = (arguments.get("query") or "").strip()
            if not query:
                return err("query required")

            top_k = min(max(int(arguments.get("top_k", TOP_K)), 1), TOP_K)
            offset = max(int(arguments.get("offset", 0)), 0)
            filename_filter = arguments.get("filename")

            results = _store.search(
                query,
                top_k=top_k,
                offset=offset,
                filename_filter=filename_filter,
            )

            if not results:
                return [
                    types.TextContent(type="text", text="No semantic records matched.")
                ]

            lines = []
            for s, m in results:
                text = (m.get("text", "") or "")[:MAX_CHARS_PER_CHUNK]
                item = f"[{m['filename']} | chunk {m['chunk']} | score {s:.3f}]\n{text}"
                lines.append(item)

            safe_lines = apply_token_budget(lines, MAX_CONTEXT_TOKENS)
            final_text = "\n\n---\n\n".join(safe_lines)

            if len(results) == top_k:
                final_text += f"\n\n[SYSTEM NOTE]: Additional context matches are available. To paginate, execute the search again with offset={offset + top_k}."

            return [types.TextContent(type="text", text=final_text)]

        return err(f"unknown tool: {name}")

    except Exception as exc:
        return err(f"Unexpected error: {exc}")


session_manager = StreamableHTTPSessionManager(
    app=server,
    event_store=None,
    json_response=True,
    stateless=True,
)


@asynccontextmanager
async def lifespan(_app):
    loop = asyncio.get_running_loop()
    ok = await loop.run_in_executor(None, _store.load)

    if not ok:
        raise RuntimeError("Failed to load LanceDB index")

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
