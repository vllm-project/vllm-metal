# Tools

## Attention Benchmark

The repository includes a local benchmark utility for comparing Metal attention backends:

```bash
source .venv-vllm-metal/bin/activate
python -m tools.benchmark.attention_benchmark
```

Running with no arguments executes the built-in `all` preset group and prints one combined text table to stdout.
By default, presets run `v1`, `v2`, `textbook`, and `sdpa`. Use `--backend all` when you also want `sdpa-compute-only`.
`num_layers` is supported as a shared benchmark setting; multi-layer runs repeat the same workload across layers and report per-layer latency.

Built-in groups:
- `all`: every built-in case
- `decode`: all decode cases
- `varlen`: all varlen cases
- `small`: `decode-small` + `varlen-light`
- `typical`: `decode-typical` + `varlen-typical`
- `long`: `decode-big-head` + `decode-long` + `varlen-single-long` + `varlen-ragged-longtail`

Built-in cases:
- `decode-small`
- `decode-typical`
- `decode-big-head`
- `decode-long`
- `varlen-light`
- `varlen-typical`
- `varlen-single-long`
- `varlen-ragged-longtail`

Useful examples:

```bash
# Run the default all group
python -m tools.benchmark.attention_benchmark

# Run a built-in group
python -m tools.benchmark.attention_benchmark --group decode
python -m tools.benchmark.attention_benchmark --group varlen
python -m tools.benchmark.attention_benchmark --group typical
python -m tools.benchmark.attention_benchmark --group long

# Run explicit cases
python -m tools.benchmark.attention_benchmark --cases decode-small,varlen-light

# Include sdpa-compute-only in addition to the default backends
python -m tools.benchmark.attention_benchmark --group all --backend all

# Write structured exports in addition to the stdout table
python -m tools.benchmark.attention_benchmark --group decode --output-json /tmp/attention.json
python -m tools.benchmark.attention_benchmark --group decode --output-csv /tmp/attention.csv

# Override shared benchmark settings on a built-in preset run
python -m tools.benchmark.attention_benchmark --group decode --num-layers 10 --iters 200

# Define a manual workload
python -m tools.benchmark.attention_benchmark --mode decode --batch-size 8 --kv-lens 2048

# Define a manual varlen workload
python -m tools.benchmark.attention_benchmark --mode varlen --q-lens 1,4,16,64 --kv-lens 128,256,512,1024
```

## Prefix Caching Benchmark

Measures TTFT / TPOT / E2EL with shared-prefix workloads using the
upstream `prefix_repetition` dataset.  Compare cache-off baseline vs
cache-on by toggling `--enable-prefix-caching` / `--no-enable-prefix-caching`.

**1. Start the server:**

```bash
# Adjust MEMORY_FRACTION based on available RAM (lower if OOM).
VLLM_METAL_USE_PAGED_ATTENTION=1 VLLM_METAL_MEMORY_FRACTION=0.7 \
  vllm serve Qwen/Qwen3-0.6B \
    --port 8000 --max-model-len 2048 --max-num-seqs 8 \
    --enable-prefix-caching
```

**2. Run the benchmark:**

```bash
vllm bench serve \
  --backend openai \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen3-0.6B \
  --dataset-name prefix_repetition \
  --num-prompts 100 \
  --prefix-repetition-prefix-len 256 \
  --prefix-repetition-suffix-len 256 \
  --prefix-repetition-num-prefixes 10 \
  --prefix-repetition-output-len 128 \
  --request-rate inf \
  --percentile-metrics ttft,tpot,e2el \
  --metric-percentiles 50,99 \
  --save-result --label cache-on
```

For a cache-off baseline, restart the server with
`--no-enable-prefix-caching` and re-run with `--label baseline`.
