#!/usr/bin/env bash
# Usage:
#   ./bench_client.sh          # paged attention OFF (default)
#   ./bench_client.sh paged    # paged attention ON

set -euo pipefail

if [[ "${1:-}" == "paged" ]]; then
    echo "=== Paged attention: ON ==="
    export VLLM_METAL_USE_PAGED_ATTENTION=1
else
    echo "=== Paged attention: OFF ==="
    export VLLM_METAL_USE_PAGED_ATTENTION=0
fi

vllm bench serve \
    --backend vllm \
    --model Qwen/Qwen3-0.6B \
    --endpoint /v1/completions \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 300 \
    --request-rate 20
