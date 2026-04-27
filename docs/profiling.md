# GPU Profiling on Apple Silicon

vllm-metal ships an in-process **frame-capture profiler** that records every Metal command issued during inference. Captures open in Xcode's GPU Frame Debugger and give you per-kernel GPU timing (via the **Profile** button), command ordering, and full buffer state — everything you typically want for both performance and correctness debugging on Apple Silicon. **This is the recommended path.**

The profiler plugs into vLLM's standard profiler API — `LLM.start_profile()`, the `POST /start_profile` HTTP endpoint, etc. — and calls `mlx.metal.start_capture` underneath. Captures must be bounded (see the [recommended recipe](#recommended-starting-recipe)).

For users already familiar with Apple's Instruments.app, Apple's `xctrace` is available as a lower-overhead, whole-process alternative for wall-clock and GPU-utilization views — see [Advanced: xctrace](#advanced-xctrace) at the end. It's a different mental model and doesn't surface MLX kernel names; reach for it only if you already know how to read Metal System Trace.

## Prerequisites

- **Full Xcode** (~10 GB) from the [Mac App Store](https://apps.apple.com/us/app/xcode/id497799835) or [developer.apple.com/xcode](https://developer.apple.com/xcode/). Apple's "Command Line Tools for Xcode" alone are **not** enough — they don't include the GPU Frame Debugger that loads `.gputrace` bundles.
- After installing Xcode, run `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer` once so command-line tools target the full Xcode install.
- `MTL_CAPTURE_ENABLED=1` must be set in the environment **before** the vLLM process starts. Apple's Metal framework reads it at process startup; setting it later has no effect.

## Quick Start

```bash
# 1. Launch vLLM with profiling configured. MTL_CAPTURE_ENABLED gates Apple's
#    frame capture; --profiler-config wires the directory into the engine.
MTL_CAPTURE_ENABLED=1 vllm serve Qwen/Qwen3-0.6B \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_dir=/tmp/metal-trace
```

```bash
# 2. From another terminal, bracket the work you want captured.
curl -X POST http://localhost:8000/start_profile
# ... send the requests you want profiled ...
curl -X POST http://localhost:8000/stop_profile
```

```bash
# 3. Open the bundle in Xcode.
open /tmp/metal-trace/*.gputrace
```

> The `--profiler-config.profiler=torch` value is mandatory for vLLM's config validation but ignored by vllm-metal — `MetalWorker.profile()` always uses Metal frame capture regardless of the value.

## Programmatic API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    profiler_config={
        "profiler": "torch",
        "torch_profiler_dir": "/tmp/metal-trace",
    },
)

llm.start_profile(profile_prefix="my_run")        # → start_capture
out = llm.generate(["Hello"], SamplingParams(max_tokens=8))
llm.stop_profile()                                # → stop_capture

# Trace lands at /tmp/metal-trace/my_run_dp0_pp0_tp0.gputrace
```

The trace filename is `<prefix>_dp<X>_pp<Y>_tp<Z>.gputrace` (data-, pipeline-, tensor-parallel ranks). On a single Apple Silicon device this resolves to `<prefix>_dp0_pp0_tp0.gputrace`.

## Opening and Reading the Trace

The `.gputrace` is an Apple bundle (a folder, not a single file). To inspect:

```bash
open /tmp/metal-trace/my_run_dp0_pp0_tp0.gputrace
```

This launches Xcode and loads the **GPU Frame Debugger**. Useful views (in the left sidebar):

| Pane | What you see | Needs profile pass? |
|------|--------------|---------------------|
| **Summary** | Command-buffer counts, total command count, high-level workload overview | No |
| **Dependencies** | Graph view of buffer-to-buffer dependencies. MLX's docs recommend this view for understanding kernel ordering. | No |
| **Performance** | Per-dispatch GPU time, memory bandwidth, occupancy. Click **Profile…** once to populate; replays the trace (a few minutes). Safe on traces from the [recommended recipe](#recommended-starting-recipe); see warning below. | **Yes** |
| **Memory** | Buffer contents and Metal heap allocations at any point in the trace. Useful for debugging numerical issues and inspecting KV cache layout. | No |
| **Frame Navigator** (the API call list) | Hierarchical list of `MTLCommandBuffer` → `MTLComputeCommandEncoder` → individual `dispatchThreadgroups` calls. The compute dispatches are nested inside the encoders — not at the top level. | No |

### Don't click Profile on default-config traces

The **Profile…** button replays the captured commands with counters enabled — fine for traces produced by the [recommended recipe](#recommended-starting-recipe) (~2 GB, replay takes a few minutes). It will lock up your machine on a default-config trace, where the engine has 20+ GiB of paged-attention KV state captured: replay tries to re-allocate all of that on the GPU at once. If Xcode beachballs after Profile, force-quit and re-run with `VLLM_METAL_MEMORY_FRACTION=0.1`.

The non-replay panes (Summary, Dependencies, Memory, Frame Navigator) work on any trace size without clicking Profile.

### Finding specific kernels

The top-level entries in Frame Navigator are MLX's heap-buffer operations (`BufferHeapOffset(0x...)`), not the compute dispatches you usually want. To find actual kernels:

- Use **Edit → Find → Find in GPU Trace** (`⌘F`) and search by kernel name: `paged_attention`, `rms_norm`, `rope`, `gemm`, `q8_0_dequant`, etc. MLX labels its dispatches with op names, so search hits them directly.
- Or change the **"Group by API Call"** dropdown above the Frame Navigator to a different grouping (e.g. by encoder) if you find the default too verbose.
- Compute dispatches always live inside an `MTLComputeCommandEncoder` — expand those to see the kernels they issued.

## Configuration

Profiler behavior is controlled via vLLM's `--profiler-config` flags (or the `profiler_config=` kwarg on `LLM`). The field vllm-metal honors:

| Field | Default | Description |
|-------|---------|-------------|
| `torch_profiler_dir` | (required) | Output directory for the `.gputrace` bundle |

The `delay_iterations` and `max_iterations` scheduling fields are **rejected** at `start_profile` time — Metal frame capture is manual start/stop only. To bound captured work, use small `max_tokens` and short prompts. Other torch-profiler-specific fields (`torch_profiler_with_stack`, `torch_profiler_record_shapes`, etc.) have no effect on the `.gputrace` output.

### Recommended starting recipe

This is the empirically-validated starting point. Smoke-tested on Qwen3-0.6B; produces a trace that finishes in ~1 minute and is small enough to inspect via Xcode's static panes (Summary, Dependencies, Memory):

```bash
# Required env — Apple's gate + our 10× KV-cache shrink.
export MTL_CAPTURE_ENABLED=1
export VLLM_METAL_MEMORY_FRACTION=0.1

# Launch the server.
vllm serve Qwen/Qwen3-0.6B \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_dir=/tmp/metal-trace
```

```python
# Or, equivalent offline-LLM form. Bound capture work with max_tokens.
from vllm import LLM, SamplingParams
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    profiler_config={"profiler": "torch", "torch_profiler_dir": "/tmp/metal-trace"},
)
llm.start_profile(profile_prefix="qwen3")
llm.generate(["Hi"], SamplingParams(max_tokens=10))
llm.stop_profile()
```

What this recipe gets you (measured numbers, Qwen3-0.6B, M-series, 10 tokens generated):

| Knob | Effect |
|---|---|
| `VLLM_METAL_MEMORY_FRACTION=0.1` | KV cache: 22.1 GB → **0.58 GB** (38× reduction) |
| `max_tokens=10` | Bounded decode work; ~10 forward passes captured |
| **Resulting trace size** | **~2.2 GB on disk** |
| **Capture wall-clock** | **~49 s** |

This trace is small enough for **all** of Xcode's panes — including Profile/replay for per-kernel timing. The KV-cache shrink (22 GB → 0.58 GB) is what makes replay feasible: the captured GPU heap state has to fit when replay re-allocates it. Without `VLLM_METAL_MEMORY_FRACTION=0.1`, Profile will lock up your machine.

## Caveats

**Capture is slow.** Apple's frame capture serializes every Metal command (kernel state, buffer contents, pipeline descriptors). For LLM inference — thousands of dispatches per token — wall-clock during capture is **50–100× slower** than uncaptured execution. Plan accordingly: don't capture entire benchmark runs.

**Traces are large.** A single forward pass through Qwen3-0.6B (28 layers) produces a ~2.6 GB `.gputrace` bundle on disk (the logical size, summed across the 30k+ captured buffers, is much higher — Metal uses sparse files). Larger models and longer captures grow accordingly. Make sure your trace dir has space.

**Xcode replay needs the KV cache shrunk.** The Profile/replay pass re-allocates all captured GPU heap state. Without `VLLM_METAL_MEMORY_FRACTION=0.1` (or smaller), the replay will try to re-allocate the engine's full ~22 GiB paged-attention cache and lock up your machine. The recommended recipe avoids this. Static panes (Summary, Dependencies, Memory) work regardless.

**`MTL_CAPTURE_ENABLED` cannot be set lazily.** Apple's framework reads it once at process startup. Our wrapper checks for it in the worker subprocess and raises a clear error if missing. If you forget to set it, you'll see:

```
RuntimeError: Metal frame capture requires MTL_CAPTURE_ENABLED=1 in the
process environment. Restart the engine with that variable set, then retry.
```

**No semantic range markers.** Unlike CUDA's NVTX or torch.profiler's `record_function`, Metal frame capture has no API for inserting named ranges. Kernel-level granularity (each MLX op shows up as a labeled Metal dispatch) is what you get; if you need per-layer or per-stage labels, those would require adding `os_signpost` calls in the C++ extension — not currently supported.

## Advanced: xctrace

For users already familiar with **Instruments.app**, Apple's `xctrace` CLI can record a whole-process trace — useful for a high-level "is the engine running healthily?" view (GPU utilization, CPU stalls, thermal state) without bounding the workload:

```bash
xctrace record --template "Metal System Trace" --output /tmp/run.trace \
  --launch -- vllm serve Qwen/Qwen3-0.6B
# Ctrl-C when done, then:
open /tmp/run.trace
```

Trace size is typically 50–150 MB regardless of run length. **Caveat**: Apple's Metal System Trace template was designed for graphics workloads, so MLX's compute kernels don't surface as named encoders in the Metal Application track. You get GPU utilization and CPU sampling, **not per-kernel timing** — for that, use frame capture's Profile pane.

