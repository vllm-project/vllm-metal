# Tools

## Greedy Parity

See the [Tools guide](../docs/tools.md) for live comparison against native `mlx-lm`.

## Prefix Caching Benchmark

Measures TTFT / TPOT / E2EL with shared-prefix workloads using the
upstream `prefix_repetition` dataset.  Compare cache-off baseline vs
cache-on by toggling `--enable-prefix-caching` / `--no-enable-prefix-caching`.

**1. Start the server:**

```bash
# Adjust --gpu-memory-utilization based on available RAM (lower if OOM).
vllm serve Qwen/Qwen3-0.6B \
    --gpu-memory-utilization 0.7 \
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

python -m tools.benchmark.gemma4_mtp_benchmark \
  --model /path/to/gemma-4-E2B-it \
  --gpu-memory-utilization 0.5 \
  --batch-size 4 --max-tokens 64 --repeats 1 --warmup 0 \
  --ignore-eos --max-model-len 1024 --max-num-batched-tokens 512 \
  --label e2b-baseline-bs4-64 \
  --output-json /tmp/gemma4-e2b-baseline-bs4-64.json

python -m tools.benchmark.gemma4_mtp_benchmark \
  --model /path/to/gemma-4-E2B-it \
  --assistant-model /path/to/gemma-4-E2B-it-assistant-bf16 \
  --gpu-memory-utilization 0.5 \
  --num-speculative-tokens 3 \
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

## DiffusionGemma serving benchmark

`diffusion_bench.py` times whole requests against a running DiffusionGemma
server and reports median and max latency and aggregate completion tokens/s.
Use it instead of the server's periodic "Avg prompt/generation throughput" log
lines, which average over a 10 s window and make a single short request look
slow. Each prompt gets an untimed warmup first, which absorbs MLX kernel
compilation and lazy weight loading.

```bash
PYTHONPATH=$PWD vllm serve mlx-community/diffusiongemma-26B-A4B-it-4bit \
  --diffusion-config '{"canvas_length": 32}'

python tools/diffusion_bench.py
python tools/diffusion_bench.py --concurrency 1 4 --repeats 3 --json bench.json
```

Tokens/s depends on how much of each 32-token canvas the answer uses: a short
reply still costs a full canvas, so compare the `long` prompt across runs, not
`short`.

## LFM2-VL vision smoke

`lfm2_vl_vision_smoke.py` checks LFM2-VL image serving against the real engine
and native mlx-vlm. It compares first-token log-probs for one image (reporting
EXACT and TOP_K_MATCH), confirms two different images do not produce the same
distribution, and runs a short greedy completion.

The second check is the important one: before image support existed the engine
accepted an image, dropped it, and answered from the text alone. That failure
is silent, so parity against a reference would not catch a regression back to
it, but two images collapsing to one distribution will.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python tools/lfm2_vl_vision_smoke.py
PYTORCH_ENABLE_MPS_FALLBACK=0 python tools/lfm2_vl_vision_smoke.py \
  --checkpoint LiquidAI/LFM2.5-VL-1.6B
```

Use a checkpoint packaged for Transformers. The MLX conversions are packaged
for mlx-vlm: the 450M declares `model_type: lfm2-vl`, which Transformers does
not register, and the 3B names `Lfm2VlNumpyImageProcessor`, which exists in no
Transformers release. Both fail before the engine starts.

Prints `SMOKE PASS` on success. The KL tolerances assume bf16 noise between the
Metal paged path and MLX's own SDPA; a failure well outside them points at the
adapter rather than at precision.
