#!/bin/sh
set -e

INDEX_DIR="/app/lancedb_index"
REPO="jonafeucht/bundesarchiv-mcp"

echo "Fetching LanceDB index from latest GitHub Release..."

DOWNLOAD_URL=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
  | jq -r '.assets[] | select(.name == "lancedb_index.tar.gz") | .browser_download_url')

if [ -z "$DOWNLOAD_URL" ] || [ "$DOWNLOAD_URL" = "null" ]; then
  echo "ERROR: Could not find lancedb_index.tar.gz in the latest release." >&2
  exit 1
fi

echo "Downloading from: $DOWNLOAD_URL"

curl -L --progress-bar -o /tmp/lancedb_index.tar.gz "$DOWNLOAD_URL"

echo "Extracting index..."

rm -rf "$INDEX_DIR"
mkdir -p "$INDEX_DIR"

tar -xzf /tmp/lancedb_index.tar.gz -C /app/

rm -f /tmp/lancedb_index.tar.gz

echo "Done."

exec python -m uvicorn main:app --host 0.0.0.0 --port 8000
