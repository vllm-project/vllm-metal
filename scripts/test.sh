#!/bin/bash

# Run a paged-attention smoke test: serve a model, check golden output.
#
# Usage: run_smoke_test <model> <revision> <prompt> <expected> [extra_serve_args...]
run_smoke_test() {
  local port="$1"
  local model="$2"
  local revision="$3"
  local prompt="$4"
  local expected="$5"
  shift 5
  local extra_args=("$@")

  section "Smoke test: $model (port $port)"

  # 1. Start vLLM with paged attention on a dedicated port
  GLOO_SOCKET_IFNAME=lo0 \
    VLLM_METAL_USE_PAGED_ATTENTION=1 \
    VLLM_METAL_MEMORY_FRACTION=0.8 \
    vllm serve "$model" --revision "$revision" --max-model-len 512 --max-num-batched-tokens 64 --port "$port" ${extra_args[@]+"${extra_args[@]}"} &

  local vllm_pid=$!

  # 2. Wait for the server to be ready
  echo "Waiting for vLLM to start..."
  local health_url="http://localhost:${port}/health"
  if ! curl --retry 30 --retry-delay 10 --retry-all-errors -s "$health_url" > /dev/null; then
    echo "vLLM failed to start."
    kill $vllm_pid
    exit 1
  fi

  echo "Model loaded successfully!"

  # 3. Test completions endpoint with golden comparison
  echo "Testing completions with golden output..."
  local response
  response=$(curl -s -X POST "http://localhost:${port}/v1/completions" \
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

  kill $vllm_pid
}

smoke_tests() {
  # Qwen3-0.6B: standard GQA paged attention path
  run_smoke_test 8100 \
    "Qwen/Qwen3-0.6B" \
    "c1899de289a04d12100db370d81485cdf75e47ca" \
    "The capital of France is" \
    " Paris. The capital of Italy is Rome. The"

  # Qwen3.5-0.8B: hybrid SDPA + GDN linear attention paged path
  # max-num-seqs=1: limits GDN linear state allocation (~10MB/slot × N slots)
  # which is critical on CI runners with only ~5GB Metal memory.
  run_smoke_test 8101 \
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
