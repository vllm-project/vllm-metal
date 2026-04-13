#!/bin/bash

# Run a paged-attention smoke test: serve a model, check golden output.
#
# Usage: run_smoke_test <model> <revision> <prompt> <expected> [extra_serve_args...]
run_smoke_test() {
  local model="$1"
  local revision="$2"
  local prompt="$3"
  local expected="$4"
  shift 4
  local extra_args=("$@")

  section "Smoke test: $model"

  # 1. Start vLLM with paged attention
  GLOO_SOCKET_IFNAME=lo0 \
    VLLM_METAL_USE_PAGED_ATTENTION=1 \
    VLLM_METAL_MEMORY_FRACTION=0.8 \
    vllm serve "$model" --revision "$revision" --max-model-len 512 --max-num-batched-tokens 64 ${extra_args[@]+"${extra_args[@]}"} &

  local vllm_pid=$!

  # 2. Wait for the server to be ready
  echo "Waiting for vLLM to start..."
  local health_url="http://localhost:8000/health"
  if ! curl --retry 30 --retry-delay 10 --retry-all-errors -s "$health_url" > /dev/null; then
    echo "vLLM failed to start."
    kill $vllm_pid
    exit 1
  fi

  echo "Model loaded successfully!"

  # 3. Test completions endpoint with golden comparison
  echo "Testing completions with golden output..."
  local response
  response=$(curl -s -X POST "http://localhost:8000/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$model\",
      \"prompt\": \"$prompt\",
      \"temperature\": 0,
      \"max_tokens\": 10
    }")

  if ! echo "$response" | grep -q '"choices"'; then
    echo "Completions test failed. Response:"
    echo "$response"
    kill $vllm_pid
    exit 1
  fi

  local actual
  actual=$(echo "$response" | python3 -c "import sys,json; print(json.loads(sys.stdin.read(), strict=False)['choices'][0]['text'])")

  if [ "$actual" != "$expected" ]; then
    echo "Golden comparison FAILED"
    echo "  expected: '$expected'"
    echo "  actual:   '$actual'"
    kill $vllm_pid
    exit 1
  fi

  echo "Smoke test passed! Output matches golden."

  # Graceful shutdown with SIGKILL fallback
  local shutdown_timeout=10
  local port_timeout=15

  kill $vllm_pid 2>/dev/null
  for ((shutdown_wait=0; shutdown_wait<shutdown_timeout; shutdown_wait++)); do
    if ! kill -0 $vllm_pid 2>/dev/null; then break; fi
    sleep 1
  done
  if kill -0 $vllm_pid 2>/dev/null; then
    echo "Warning: vLLM did not terminate after ${shutdown_timeout}s, sending SIGKILL"
    kill -9 $vllm_pid 2>/dev/null
  fi
  wait $vllm_pid 2>/dev/null || true

  # Wait for port to be released before next test
  local port_wait=0
  while lsof -i :8000 -sTCP:LISTEN >/dev/null 2>&1; do
    sleep 1
    port_wait=$((port_wait + 1))
    if [ $port_wait -ge $port_timeout ]; then
      echo "Warning: port 8000 still in use after ${port_timeout}s"
      break
    fi
  done
}

smoke_tests() {
  # Qwen3-0.6B: standard GQA paged attention path
  run_smoke_test \
    "Qwen/Qwen3-0.6B" \
    "c1899de289a04d12100db370d81485cdf75e47ca" \
    "The capital of France is" \
    " Paris. The capital of Italy is Rome. The"

  # Qwen3.5-0.8B: hybrid SDPA + GDN linear attention paged path
  # max-num-seqs=1: limits GDN linear state allocation (~10MB/slot × N slots)
  # which is critical on CI runners with only ~5GB Metal memory.
  run_smoke_test \
    "Qwen/Qwen3.5-0.8B" \
    "2fc06364715b967f1860aea9cf38778875588b17" \
    "The capital of France is" \
    " Paris.
The capital of France is Paris." \
    --max-num-seqs 1
}

installs() {
  section "Installing vllm"

  if [ "$(uname)" == "Darwin" ]; then
    ./install.sh
  fi
}

main() {
  set -eu -o pipefail

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  # shellcheck source=lib.sh disable=SC1091
  source "${script_dir}/lib.sh"

  setup_dev_env

  if [ "$(uname)" == "Darwin" ]; then
    installs
    # shellcheck source=/dev/null
    source .venv-vllm-metal/bin/activate

    section "Verifying package import"
    python -c "import vllm_metal; print('vllm_metal imported successfully')"

    smoke_tests
    section "Running tests"
    # Exclude perf/long-running tests by default; run them explicitly via:
    #   pytest -m slow tests/ -v --tb=short
    pytest -m "not slow" tests/ -v --tb=short
  fi
}

main "$@"
