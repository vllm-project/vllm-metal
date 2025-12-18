#!/bin/bash

smoke_test() {
  ./install.sh

  # 1. Start vLLM in the background
  vllm serve Qwen/Qwen3-0.6B
    
  # Store the process ID
  local vllm_pid=$!

  # 2. Wait for the server to be ready
  echo "Waiting for vLLM to start..."
  local url="http://localhost:8000/health"
  if ! curl --retry 8 --retry-all-errors -s "$url" > /dev/null; then
    echo "vLLM failed to start."

    kill $vllm_pid

    exit 1
  fi

  echo "Model loaded successfully!"

  kill $vllm_pid
}

installs() {
  # Install vllm with --no-deps to avoid CUDA dependencies and install its macOS-compatible deps
  section "Installing vllm"
  uv pip install --upgrade --no-deps vllm
  # Install minimal deps needed for vllm to import (for tests)
  uv pip install pydantic cbor2 msgspec cloudpickle prometheus-client fastapi uvicorn uvloop pillow \
    tiktoken typing_extensions filelock py-cpuinfo aiohttp openai einops importlib_metadata mistral_common \
    pyyaml requests tqdm sentencepiece gguf blake3 pyzmq regex protobuf setuptools depyf numba \
    tokenizers cachetools

  if is_apple_silicon; then
    brew install shellcheck
  fi
}

linters() {
  section "Running shellcheck"
  shellcheck -- *.sh scripts/*.sh

  section "Running ruff linter"
  ruff check .

  section "Running ruff formatter check"
  ruff format --check .

  section "Running mypy type checker"
  mypy vllm_metal
}

main() {
  set -eu -o pipefail

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  # shellcheck source=lib.sh disable=SC1091
  source "${script_dir}/lib.sh"

  setup_dev_env

  installs

  linters

  section "Running tests"
  pytest tests/ -v --tb=short

  section "Verifying package import"
  python -c "import vllm_metal; print('vllm_metal imported successfully')"

  smoke_test
}

main "$@"
