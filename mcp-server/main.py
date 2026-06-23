import asyncio
import os
from contextlib import asynccontextmanager

import lancedb
import uvicorn
from dotenv import load_dotenv
from mcp import types
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from sentence_transformers import SentenceTransformer
from src.middleware.api_key import APIKeyMiddleware
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Mount

load_dotenv()

DB_URI = os.environ.get("DB_URI", "./lancedb_index")
TABLE_NAME = "document_chunks"
TOP_K = int(os.environ.get("TOP_K", 8))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")


class VectorStore:
    """Search engine utilizing an embedded LanceDB table."""

    def __init__(self, embed_model: str, db_uri: str) -> None:
        self._model = SentenceTransformer(embed_model)
        self._db_uri = db_uri
        self._db = None
        self._table = None

    def load(self) -> bool:
        try:
            self._db = lancedb.connect(self._db_uri)

            tables = self._db.table_names()

            if TABLE_NAME not in tables:
                print(f"Error: Table '{TABLE_NAME}' missing at {self._db_uri}")
                return False

            self._table = self._db.open_table(TABLE_NAME)

            print(f"Ready: LanceDB connected. Table size={len(self._table)}")
            return True

        except Exception as e:
            print(f"Failed to load LanceDB: {e}")
            return False

    def list_pdfs(self) -> list[str]:
        if self._table is None:
            return []

        df = self._table.to_pandas()

        if "filename" in df.columns:
            return sorted(df["filename"].dropna().unique().tolist())

        return []

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        filename_filter: str | None = None,
    ) -> list[tuple[float, dict]]:
        if self._table is None:
            return []

        query_vec = self._model.encode(query, normalize_embeddings=True).tolist()

        qb = self._table.search(query_vec).limit(top_k)

        if filename_filter:
            qb = qb.where(f"filename = '{filename_filter}'")

        df = qb.to_pandas()

        results: list[tuple[float, dict]] = []

        for _, row in df.iterrows():
            score = float(row.get("_distance", row.get("_score", 0.0)))

            meta = {
                "filename": row["filename"],
                "chunk": int(row["chunk"]),
                "text": row["text"],
            }

            results.append((score, meta))

        return results


_store = VectorStore(EMBED_MODEL, DB_URI)
server = Server("Akten für das Bundesarchiv")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="list_pdfs",
            description="List all available indexed files.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="search",
            description=(
                "Semantic search across all contexts. Returns relevant passages."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filename": {
                        "type": "string",
                        "description": "Optional filter by filename",
                    },
                    "top_k": {"type": "integer"},
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
            pdfs = _store.list_pdfs()
            text = "\n".join(pdfs) if pdfs else "(no source metadata available)"
            return [types.TextContent(type="text", text=text)]

        elif name == "search":
            query = (arguments.get("query") or "").strip()
            if not query:
                return err("query required")

            top_k = int(arguments.get("top_k", TOP_K))
            filename_filter = arguments.get("filename")

            results = _store.search(query, top_k=top_k, filename_filter=filename_filter)

            if not results:
                return [
                    types.TextContent(type="text", text="No semantic records matched.")
                ]

            lines = [
                f"[{m['filename']} | chunk {m['chunk']} | score {s:.3f}]\n{m['text']}"
                for s, m in results
            ]

            return [types.TextContent(type="text", text="\n\n---\n\n".join(lines))]

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
