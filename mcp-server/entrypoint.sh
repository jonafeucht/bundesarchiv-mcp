#!/bin/sh
set -e

INDEX_DIR="/app/lancedb_index"
REPO="jonafeucht/bundesarchiv-mcp"

if [ ! -d "$INDEX_DIR" ] || [ -z "$(ls -A "$INDEX_DIR" 2>/dev/null)" ]; then
  echo "LanceDB index not found. Fetching from latest GitHub Release..."

  DOWNLOAD_URL=$(curl -fsSL \
    "https://api.github.com/repos/${REPO}/releases/latest" \
    | grep -o '"browser_download_url": *"[^"]*lancedb_index\.tar\.gz"' \
    | grep -o 'https://[^"]*')

  if [ -z "$DOWNLOAD_URL" ]; then
    echo "ERROR: Could not find lancedb_index.tar.gz in the latest release." >&2
    exit 1
  fi

  echo "Downloading from: $DOWNLOAD_URL"
  curl -fsSL -o /tmp/lancedb_index.tar.gz "$DOWNLOAD_URL"

  mkdir -p "$INDEX_DIR"
  tar -xzf /tmp/lancedb_index.tar.gz -C /app/
  rm /tmp/lancedb_index.tar.gz
  echo "Done."
else
  echo "Index already present, skipping download."
fi

exec python -m uvicorn main:app --host 0.0.0.0 --port 8000
