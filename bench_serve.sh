#!/usr/bin/env bash
# Usage:
#   ./bench_serve.sh          # paged attention OFF (default)
#   ./bench_serve.sh paged    # paged attention ON

set -euo pipefail

if [[ "${1:-}" == "paged" ]]; then
    echo "=== Paged attention: ON ==="
    export VLLM_METAL_USE_PAGED_ATTENTION=1
else
    echo "=== Paged attention: OFF ==="
    export VLLM_METAL_USE_PAGED_ATTENTION=0
fi

vllm serve Qwen/Qwen3-0.6B --max-model-len 2048
