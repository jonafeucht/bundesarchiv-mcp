# MCP Server providing semantic retrieval

An MCP Server providing semantic retrieval across a curated archive of historical documents from the Bundesarchiv, CIA FOIA, the Library of Congress, and more.

## Configuration

Copy `.env.example` to `.env` and adjust values:

```env
MCP_API_KEY=your-secret-key
USE_API_KEY=true
CHUNK_SIZE=750
CHUNK_OVERLAP=100
TOP_K=10
MAX_CHARS_PER_CHUNK=6000
MAX_CONTEXT_TOKENS=24000
EMBED_MODEL=all-MiniLM-L6-v2
```

## Deploy

```yml
services:
  bundesarchiv-mcp:
    image: ghcr.io/jonafeucht/bundesarchiv-mcp:latest
    ports:
      - "8000:8000"
    env_file:
      - .env
    restart: unless-stopped
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
