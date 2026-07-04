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

## Gemma4 MTP Benchmark

Compares a Gemma4 target-only baseline with the same target plus a Gemma4 MTP
assistant. Run one mode per process so model state does not leak between runs:

```bash
source .venv-vllm-metal/bin/activate
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_METAL_MEMORY_FRACTION=0.5

python -m tools.benchmark.gemma4_mtp_benchmark \
  --model /path/to/gemma-4-E2B-it \
  --batch-size 4 --max-tokens 64 --repeats 1 --warmup 0 \
  --ignore-eos --max-model-len 1024 --max-num-batched-tokens 512 \
  --label e2b-baseline-bs4-64 \
  --output-json /tmp/gemma4-e2b-baseline-bs4-64.json

python -m tools.benchmark.gemma4_mtp_benchmark \
  --model /path/to/gemma-4-E2B-it \
  --assistant-model /path/to/gemma-4-E2B-it-assistant-bf16 \
  --batch-size 4 --max-tokens 64 --repeats 1 --warmup 0 \
  --ignore-eos --max-model-len 1024 --max-num-batched-tokens 512 \
  --label e2b-mtp-bs4-64 \
  --output-json /tmp/gemma4-e2b-mtp-bs4-64.json
```

The output JSON includes package versions, relevant environment variables,
prompts, generated token IDs, elapsed time, and output tokens per second.

## Spec-decode eval dataset

Speculative decoding must be evaluated on *natural* prompts — acceptance rate (and
therefore speedup) is much lower on synthetic sets like `sonnet`/`random`.
`build_spec_bench_dataset.py` downloads `RedHatAI/speculator_benchmarks` (the dataset
the vLLM `speculators` repo benchmarks with) and writes a single `spec_bench`-format
file (`turns` column) that vLLM's `--dataset-name spec_bench` loader reads directly,
length-filtering prompts to fit the target context and sampling a fixed number per
category. To change the sample (dataset, per-category count, length budget, seed), edit
the constants at the top of the script.

```bash
# Build the eval set -> spec_bench_sample.jsonl
python tools/build_spec_bench_dataset.py

# Benchmark speculative decoding (greedy is required for drafting to engage)
vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-8B --endpoint /v1/completions \
  --dataset-name spec_bench --dataset-path spec_bench_sample.jsonl --spec-bench-output-len 128 \
  --num-prompts 100 --request-rate 10 --max-concurrency 32 \
  --temperature 0 --top-p 1.0 --top-k -1 --ignore-eos --seed 0
```
