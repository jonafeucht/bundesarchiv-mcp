# Akten der Reichskanzlei – Bundesarchiv MCP

MCP server for searching and extracting text from Bundesarchiv PDF documents, served over Streamable HTTP.

## Configuration

Copy `.env.example` to `.env` and adjust values:

```env
MCP_API_KEY=your-secret-key
USE_API_KEY=false
INDEX_PATH=./faiss_index
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=50
EMBED_MODEL=all-MiniLM-L6-v2
```

## Integration

```json
{
  "mcpServers": {
    "bundesarchiv-mcp": {
      "url": "http://localhost:8000/mcp",
      "headers": {
        "x-api-key": "your-secret-key"
      }
    }
  }
}
```
