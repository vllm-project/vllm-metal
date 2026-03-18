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
