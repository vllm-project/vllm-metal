#!/bin/bash

smoke_test() {
  ./install.sh

  # 1. Start vLLM in the background
  vllm serve Qwen/Qwen3-0.6B &
    
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
  section "Installing vllm dependencies"

  # Install vllm dependencies first (macOS-compatible packages)
  uv pip install pydantic cbor2 msgspec cloudpickle prometheus-client fastapi uvicorn uvloop pillow \
    tiktoken typing_extensions filelock py-cpuinfo aiohttp openai einops importlib_metadata mistral_common \
    pyyaml requests tqdm sentencepiece gguf blake3 pyzmq regex protobuf setuptools depyf numba \
    tokenizers cachetools partial-json-parser compressed-tensors torch transformers accelerate safetensors

  section "Installing vllm"

  # Try to install vllm - first from wheel, then from source
  VLLM_INSTALLED=false

  if uv pip install --only-binary=:all: vllm 2>/dev/null; then
    VLLM_INSTALLED=true
    echo "vLLM installed from wheel"
  elif is_apple_silicon; then
    # On Apple Silicon, vllm wheels don't exist and source build requires Intel MKL
    # which doesn't work on ARM. We can still run tests using the _compat.py fallbacks.
    echo "WARNING: vLLM cannot be installed on macOS Apple Silicon (no wheels, source build requires Intel MKL)"
    echo "Tests will run with fallback stubs from _compat.py"
    echo "smoke_test will be skipped"
  else
    echo "No pre-built vLLM wheel available, building from source..."

    # Ensure cmake is available
    if ! command -v cmake &> /dev/null; then
      if command -v brew &> /dev/null; then
        brew install cmake ninja || true
      fi
    fi

    if command -v cmake &> /dev/null; then
      # Build vLLM from source with CPU target (no CUDA)
      if VLLM_TARGET_DEVICE=cpu MAX_JOBS=4 uv pip install vllm 2>&1; then
        VLLM_INSTALLED=true
        echo "vLLM built from source successfully"
      fi
    fi

    if [ "$VLLM_INSTALLED" = false ]; then
      error "Failed to install vLLM. vLLM is a mandatory dependency."
      echo "Try installing cmake: brew install cmake ninja"
      exit 1
    fi
  fi

  # Verify vLLM is importable (if installed)
  if [ "$VLLM_INSTALLED" = true ]; then
    if ! python -c "import vllm" 2>/dev/null; then
      error "vLLM installed but not importable"
      exit 1
    fi
  fi

  if is_apple_silicon; then
    if ! command -v shellcheck &> /dev/null; then
      brew install shellcheck
    fi
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

  # Global variable set by installs()
  VLLM_INSTALLED=false

  setup_dev_env

  installs

  linters

  section "Running tests"
  pytest tests/ -v --tb=short

  section "Verifying package import"
  python -c "import vllm_metal; print('vllm_metal imported successfully')"

  if [ "$VLLM_INSTALLED" = true ]; then
    smoke_test
  else
    section "Skipping smoke_test (vLLM not installed)"
    echo "smoke_test requires vLLM to be installed"
  fi
}

main "$@"
