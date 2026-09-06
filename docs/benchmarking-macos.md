# Benchmarking vllm-metal on macOS

Three measurement hazards documented in [#713] can silently distort
performance numbers on Apple Silicon. All three were hit while
validating the decode work in #693 on an M5 Pro; each cost hours of
misattribution before being isolated.

## 1. Absolute numbers: use the server topology

In-process engines (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, i.e. the `LLM(...)`
offline API) run the engine core as a thread sharing one interpreter/GIL
with the driver's output processing. At long-context decode this
understates absolute throughput by **1.5-2.1x** versus `vllm serve`, and
it **compresses ratios** (a true 2.22x comparison read as 1.56x), because
the penalty grows with the speed of the path measured.

Rules:

- Absolute performance claims: server topology only.
- In-process is for debugging and, at most, same-harness A/B with an
  explicit label that both legs share the penalty.
- To A/B a dispatch gate under the server topology, force the gate with
  an env instead of monkeypatching in-process, e.g.
  `VLLM_METAL_DISABLE_GQA_DECODE=1` (mirrors `VLLM_METAL_DISABLE_NAX`).

## 2. Check GPU co-tenancy before every measurement window

WindowServer and app renderers (browsers, Electron-style clients) can
hold the GPU at 50%+ device utilization while every vLLM process is
idle. A workload then time-shares the GPU with the compositor: measured
impact was **-16% for real decode** and up to 5x for short probes. It
mimics a machine regression and it is neither (thermal logs stay clean).

```bash
ioreg -r -c AGXAccelerator -d "1" | grep -oE '"Device Utilization %"=[0-9]+'
```

- `<= 12%` sustained: quiet window, absolute numbers meaningful.
- Higher: co-tenant active. Label results as "realistic desktop load"
  or wait; do not treat as a regression.

Record this utilization alongside every benchmark you publish.

## 3. Short-burst GEMM probes are not machine-state diagnostics

A fresh-process matmul "canary" reads the **parked-clock rate** (~25
TFLOPS on an M5 Pro where the sustained rate is ~120) until some heavier
workload has run; short bursts neither trigger nor benefit from the
clock ramp. A 2048^3 probe additionally caps at ~19-30 TFLOPS
regardless of state - the shape never reaches the fast GEMM path.

- Measure machine capability with the real workload, or with a
  **sustained (>= 30 s) probe**.
- Never use short-probe TFLOPS as a fast/slow verdict; the utilization
  gate in section 2 is the reliable co-tenancy signal.

[#713]: https://github.com/vllm-project/vllm-metal/issues/713
