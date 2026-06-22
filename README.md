# Akten der Reichskanzlei – Bundesarchiv MCP

MCP server for searching and extracting text from Bundesarchiv PDF documents, served over Streamable HTTP.

## Configuration

Copy `.env.example` to `.env` and adjust values:

```env
MCP_API_KEY=your-secret-key
USE_API_KEY=true
```

| Variable | Description | Default |
|---|---|---|
| `MCP_API_KEY` | API key for authenticating requests | *(none)* |
| `USE_API_KEY` | Enable/disable API key auth | `true` |

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

## Tools

| Tool | Description |
|---|---|
| `list_pdfs` | List all available PDF files in the configured directory |
| `get_pdf_text` | Return the full extracted text of a given PDF |
| `search_pdf` | Search for a query string inside a PDF and return matching lines (up to 50) |
