# Benchmarking on macOS

Use a real `vllm serve` process for serving performance claims, with the
same workload and configuration in each comparison. Kernel timing, offline
generation and HTTP serving answer different questions; report them separately.
The measurements in [#713](https://github.com/vllm-project/vllm-metal/issues/713)
illustrate how process topology, desktop GPU activity and short probes can
distort both absolute numbers and speedup ratios.

## Start with a serving workload

Activate the [project environment](installation.md). For source changes,
follow the [native build instructions](CONTRIBUTING.md#editing-the-metal-kernels)
and restart the server so it loads the tested Python code and native artifacts.

This example measures one request at a time with a local checkpoint. Choose
lengths that fit the model and available KV cache; 8K is an example workload,
not a guarantee that a particular attention optimization is eligible.

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=1 vllm serve /path/to/model \
  --host 127.0.0.1 --port 8000 --served-model-name bench \
  --max-model-len 9216 --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --no-enable-prefix-caching --generation-config vllm
```

From another terminal in the same environment:

```bash
vllm bench serve --model /path/to/model --served-model-name bench \
  --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
  --dataset-name random --random-input-len 8192 --random-output-len 128 \
  --random-range-ratio 0 --num-prompts 8 --num-warmups 2 \
  --request-rate inf --max-concurrency 1 --ignore-eos --temperature 0 --seed 713 \
  --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,95,99 \
  --save-result --save-detailed --result-dir benchmark-results
```

Use [`vllm bench serve`](https://docs.vllm.ai/en/latest/cli/bench/serve/) for
other workloads. Increase both the server's sequence limit and the client's
concurrency for concurrent serving. Client concurrency does not establish the
number of decode rows in an actual scheduler step. Synthetic prompts help
control workload size; use representative prompts as well when content affects
the work, particularly speculative acceptance or MoE routing.

Setting `VLLM_ENABLE_V1_MULTIPROCESSING=0` changes the engine process topology
and can change CPU/GPU overlap and throughput. `LLM(...)` does not itself imply
that this setting is off. Offline measurements remain useful when labeled with
their actual topology, but their absolute rates or A/B ratios should not be
substituted for serving results.

## Control warmup and machine state

- Warm every comparison arm with the real workload. Exclude model loading,
  compilation and discarded warmup requests from steady-state timings; measure
  cold start separately if it matters. The two warmup requests above are a
  starting point, not proof that clocks, caches and timings have stabilized.
- Repeat A/B runs in alternating or randomized order. Keep power source and
  power mode consistent; record background GPU activity before and between
  measurement windows, along with observations during the workload.
- WindowServer, browsers and other applications can use the GPU even when
  vLLM is idle. For isolated measurements, reduce competing activity. To study
  desktop contention, keep that load controlled and label the results.
- A short GEMM probe also measures clock ramp-up, launch overhead and its
  particular matrix shape. Prefer sustained real inference when assessing
  performance. A longer probe (for example, 30 seconds) is a diagnostic aid,
  not a universal GPU-health test or a replacement for the serving workload.

Useful read-only observations on macOS:

```bash
pmset -g batt
pmset -g custom
pmset -g therm
ioreg -r -c AGXAccelerator -d 1 | grep -oE '"Device Utilization %"=[0-9]+'
```

Sample repeatedly rather than relying on one reading. IORegistry counters vary
by device and macOS version; missing output means unavailable, not zero usage.
The utilization counter is device-wide: during inference it includes the
benchmark itself and cannot attribute usage to a competing process. There is
no universal utilization cutoff that certifies a clean comparison. Likewise,
no thermal warning does not prove stable GPU clocks. Record memory pressure
and swapping if the workload approaches available unified memory.

## Compare the same work and verify the route

Keep model weights and quantization, tokenizer, input token IDs, output length,
sampling settings, cache dtype, context limits, prefill chunk size and
concurrency fixed. Record actual input/output token counts and the timed KV
length range; requested lengths alone may not describe the executed workload.
Disable prefix caching for uncached comparisons, as above, or deliberately
control the cached prefixes in both arms.

For a [GQA decode](gqa-decode.md) comparison, start one server with
`VLLM_METAL_DISABLE_GQA_DECODE=0` and a separate server with
`VLLM_METAL_DISABLE_GQA_DECODE=1`, keeping all other settings identical. Run
them sequentially and stop the previous server and its workers before starting
the next arm. Zero allows normal selection; it does not force GQA or a specific
partition, and both arms can legitimately use the same fallback.

Verify the executed path in the worker that runs attention. For GQA, follow
the [dispatch diagnostic procedure](gqa-decode.md#validation): opt in while
evaluation is idle, evaluate the operation, then read the recorded family and
partition. A selector query is not execution evidence. The getters record the
last process-wide dispatch, not a per-request or per-layer trace; under
concurrency, use a separate controlled route check or a suitably scoped trace.
Keep diagnostics consistent between arms and perform expensive
[GPU captures](profiling.md) separately from throughput measurements.

Numerical attention tests, actual dispatch and serving speed are separate
evidence. Natural generations can diverge even with greedy sampling; for
content-sensitive comparisons, use a controlled token history and label that
protocol. Neither close attention outputs nor matching throughput establishes
model-quality equivalence.

## Interpret and report the result

| Metric | What it measures |
| --- | --- |
| TTFT | Client time to the first generated token, including queueing and prefill. HTTP response headers alone are not that token. |
| TPOT / inter-token latency | Generation timing after the first token; retain the tool's definition and report the distribution. |
| Output throughput | Completed output tokens divided by the benchmark duration, including the workload's prefill and queueing. It is not a decode-only rate. |
| Kernel time | The measured attention operation, excluding the rest of the model and serving stack. |

A custom streaming client must count returned token IDs or completion usage,
not SSE chunks, characters or words. Chunks can contain multiple tokens. For
a per-request decode-only rate, use token counts and timestamps from the same
window, excluding tokens already received in its first event; retain the
chunking details rather than treating each event as one token.

Report repeated measurements with spread and paired comparisons where possible,
not just the fastest run. Preserve near-flat, noisy and order-sensitive results;
do not assume a small loss is thermal noise without evidence. Save detailed
results and server logs with:

- Hardware (chip, GPU cores, RAM), macOS, power mode and observed competing load.
- vLLM, vllm-metal and MLX versions; source revision and build mode for local code.
- Model/checkpoint identity, relevant settings, commands, actual token lengths,
  warmup procedure, repetitions and ordering.
- Executed routes, successful/failed request counts, latency distributions and
  the definition of each reported speedup.

Attach only the scope, final results and material limitations to the PR;
link the detailed records so others can reproduce the comparison.
