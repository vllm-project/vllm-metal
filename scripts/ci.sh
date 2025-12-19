#!/bin/bash

smoke_test() {
  section "Running smoke test"

  # 1. Start vLLM in the background
  vllm serve Qwen/Qwen3-0.6B &

  # Store the process ID
  local vllm_pid=$!

  # 2. Wait for the server to be ready
  echo "Waiting for vLLM to start..."
  local health_url="http://localhost:8000/health"
  if ! curl --retry 8 --retry-all-errors -s "$health_url" > /dev/null; then
    echo "vLLM failed to start."

    kill $vllm_pid

    exit 1
  fi

  echo "Model loaded successfully!"

  # 3. Test chat completions endpoint
  echo "Testing chat completions endpoint..."
  local chat_url="http://localhost:8000/v1/chat/completions"
  local response
  response=$(curl -s -X POST "$chat_url" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen3-0.6B",
      "messages": [{"role": "user", "content": "Say hello"}],
      "max_tokens": 32
    }')

  if ! echo "$response" | grep -q '"choices"'; then
    echo "Chat completions test failed. Response:"
    echo "$response"
    kill $vllm_pid
    exit 1
  fi

  echo "Chat completions test passed!"

  kill $vllm_pid
}

installs() {
  section "Installing vllm"

  if [ "$(uname)" == "Darwin" ]; then
    ./install.sh
  fi

  if is_apple_silicon; then
    if ! command -v shellcheck &> /dev/null; then
      brew install shellcheck
    fi

    if ! command -v cmake &> /dev/null; then
      brew install cmake
    fi

    if ! command -v ninja &> /dev/null; then
      brew install ninja
    fi

    if ! command -v ruff &> /dev/null; then
      brew install ruff
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

  setup_dev_env

  installs
  # shellcheck source=/dev/null
  source .venv-vllm-metal/bin/activate

  linters

  if [ "$(uname)" == "Darwin" ]; then
    section "Running tests"
    pytest tests/ -v --tb=short

    section "Verifying package import"
    python -c "import vllm_metal; print('vllm_metal imported successfully')"

    smoke_test
  fi
}

main "$@"
