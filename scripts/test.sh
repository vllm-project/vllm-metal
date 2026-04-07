#!/bin/bash

smoke_test() {
  section "Running smoke test"

  local model="Qwen/Qwen3-0.6B"
  local revision="c1899de289a04d12100db370d81485cdf75e47ca"
  local prompt="The capital of France is"
  local expected=" Paris. The capital of Italy is Rome. The"

  # 1. Start vLLM with paged attention (GQA path)
  GLOO_SOCKET_IFNAME=lo0 \
    VLLM_METAL_USE_PAGED_ATTENTION=1 \
    VLLM_METAL_MEMORY_FRACTION=0.5 \
    vllm serve "$model" --revision "$revision" --max-model-len 512 &

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

  # 3. Test completions endpoint with golden comparison
  echo "Testing completions with golden output..."
  local completions_url="http://localhost:8000/v1/completions"
  local response
  response=$(curl -s -X POST "$completions_url" \
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

smoke_test_qwen35() {
  section "Running Qwen3.5 hybrid smoke test"

  local model="Qwen/Qwen3.5-0.8B"
  local revision="2fc06364715b967f1860aea9cf38778875588b17"
  local prompt="The capital of France is"
  local expected=" Paris.
The capital of France is Paris."

  # 1. Start vLLM with paged attention (hybrid SDPA + GDN path)
  GLOO_SOCKET_IFNAME=lo0 \
    VLLM_METAL_USE_PAGED_ATTENTION=1 \
    VLLM_METAL_MEMORY_FRACTION=0.5 \
    vllm serve "$model" --revision "$revision" --max-model-len 512 &

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
  local completions_url="http://localhost:8000/v1/completions"
  local response
  response=$(curl -s -X POST "$completions_url" \
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

  echo "Qwen3.5 smoke test passed! Output matches golden."

  kill $vllm_pid
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

    smoke_test
    smoke_test_qwen35
    section "Running tests"
    # Exclude perf/long-running tests by default; run them explicitly via:
    #   pytest -m slow tests/ -v --tb=short
    pytest -m "not slow" tests/ -v --tb=short
  fi
}

main "$@"
