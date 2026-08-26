#!/usr/bin/env bash
set -euo pipefail

# Manual foreground launcher for the TEI BGE reranker.
# Keep these arguments aligned with the RAG_RERANK_* values in .env.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

exec text-embeddings-router \
  --model-id BAAI/bge-reranker-v2-m3 \
  --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e \
  --hostname 127.0.0.1 \
  --port 8081 \
  --huggingface-hub-cache "$PROJECT_ROOT/data/models/huggingface" \
  --max-client-batch-size 16 \
  --max-batch-requests 1
