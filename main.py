import os
from pathlib import Path

import fitz
import uvicorn
from dotenv import load_dotenv
from mcp import types
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Mount

from src.middleware.api_key import APIKeyMiddleware

load_dotenv()

PDF_DIR = Path(os.environ.get("PDF_DIR", "./pdfs"))


def extract_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


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
            name="get_pdf_text",
            description="Return full text of a PDF.",
            inputSchema={
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
            },
        ),
        types.Tool(
            name="search_pdf",
            description="Search for text inside a PDF.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["filename", "query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "list_pdfs":
        result = [str(f.relative_to(PDF_DIR)) for f in PDF_DIR.rglob("*.pdf")]
        return [types.TextContent(type="text", text="\n".join(result))]

    elif name == "get_pdf_text":
        pdf = PDF_DIR / arguments["filename"]
        text = extract_text(pdf) if pdf.exists() else "PDF not found"
        return [types.TextContent(type="text", text=text)]

    elif name == "search_pdf":
        pdf = PDF_DIR / arguments["filename"]
        if not pdf.exists():
            return [types.TextContent(type="text", text="PDF not found")]
        text = extract_text(pdf)
        matches = [
            l for l in text.splitlines() if arguments["query"].lower() in l.lower()
        ]
        return [types.TextContent(type="text", text="\n".join(matches[:50]))]

    raise ValueError(f"Unknown tool: {name}")


session_manager = StreamableHTTPSessionManager(
    app=server,
    event_store=None,
    json_response=True,
    stateless=True,
)


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
    routes=[Mount("/mcp", app=handle_mcp)],
    lifespan=lambda app: session_manager.run(),
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
