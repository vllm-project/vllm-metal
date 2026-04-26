# GPU Profiling on Apple Silicon

vllm-metal supports two complementary profiling paths on Apple Silicon. **Pick the right one for your goal:**

| If you want… | Use | Output |
|---|---|---|
| Per-kernel **timing**, GPU utilization, CPU/GPU correlation | `xctrace record --template "Metal System Trace"` | `.trace` bundle, opens in **Instruments.app** |
| Captured **command list**, kernel dependencies, buffer state inspection (correctness debugging) | The plugin's `MetalProfilerWrapper` (start/stop frame capture) | `.gputrace` bundle, opens in **Xcode** |

**In short: for "how fast?" use xctrace. For "what ran, in what order, on what state?" use frame capture.** The two tools are not interchangeable — frame capture's per-kernel timing requires a *replay pass* that does not scale to LLM workloads (replaying 30,000 dispatches with 20+ GiB of buffer state will lock up your machine). Performance-counter analysis on Apple Silicon for inference-scale workloads should always go through `xctrace`, not frame-capture replay.

The plugin only ships an in-process driver for the **frame capture** path — `xctrace` is a system tool that wraps the whole process from outside, so it needs no plugin support. See [System-wide tracing](#system-wide-tracing-with-instruments) for the recommended xctrace recipe.

## Frame capture (in-process)

The plugin's `MetalProfilerWrapper` plugs into vLLM's existing profiler abstraction, so the standard `LLM.start_profile()` / `LLM.stop_profile()` API and the `POST /start_profile` / `POST /stop_profile` HTTP endpoints both route through it without modification. Underneath, it calls `mlx.metal.start_capture` / `stop_capture` (Apple's `MTLCaptureManager`).

**Use this for correctness debugging**, not timing. See [When frame capture's replay hits a wall](#when-frame-captures-replay-hits-a-wall) below for the size-limit caveat.

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
| **Performance** | Per-dispatch GPU time, memory bandwidth, occupancy. | **Yes — see below** |
| **Memory** | Buffer contents and Metal heap allocations at any point in the trace. Useful for debugging numerical issues and inspecting KV cache layout. | No |
| **Frame Navigator** (the API call list) | Hierarchical list of `MTLCommandBuffer` → `MTLComputeCommandEncoder` → individual `dispatchThreadgroups` calls. The compute dispatches are nested inside the encoders — not at the top level. | No |

### When frame capture's replay hits a wall

When you click **Performance** on a freshly opened `.gputrace`, Xcode shows *"Performance data not available"* with a **Profile…** button. The button **replays** the captured commands on your GPU with performance counters on. For game-style frames (a few hundred dispatches, ~100 MB of buffer state) this completes in seconds. **For LLM inference, it will not complete.** A typical capture from this plugin is 30,000+ dispatches and 20+ GiB of buffer state — replay has to re-allocate all of that on the GPU and re-execute every kernel sequentially with counter sampling. Your machine will run out of memory and lock up before the replay finishes.

**Don't click Profile… on full-engine traces.** Either:

- Use the [system-wide xctrace path](#system-wide-tracing-with-instruments) for timing — that's what it's for.
- Or, if you specifically need replay, [shrink the trace first](#shrinking-the-frame-capture-window) so it fits.

The non-replay panes (**Summary**, **Dependencies**, **Memory**, and the Frame Navigator) all work without ever clicking Profile…. They give you the captured command graph, kernel order, and buffer state immediately.

### Finding specific kernels

The top-level entries in Frame Navigator are MLX's heap-buffer operations (`BufferHeapOffset(0x...)`), not the compute dispatches you usually want. To find actual kernels:

- Use **Edit → Find → Find in GPU Trace** (`⌘F`) and search by kernel name: `paged_attention`, `rms_norm`, `rope`, `gemm`, `q8_0_dequant`, etc. MLX labels its dispatches with op names, so search hits them directly.
- Or change the **"Group by API Call"** dropdown above the Frame Navigator to a different grouping (e.g. by encoder) if you find the default too verbose.
- Compute dispatches always live inside an `MTLComputeCommandEncoder` — expand those to see the kernels they issued.

For a system-wide picture (CPU side + GPU side together, MLX scheduling, dispatch latency), use Instruments instead — see [System-wide tracing](#system-wide-tracing-with-instruments) below.

## Configuration

Profiler behavior is controlled via vLLM's `--profiler-config` flags (or the `profiler_config=` kwarg on `LLM`). The field vllm-metal honors:

| Field | Default | Description |
|-------|---------|-------------|
| `torch_profiler_dir` | (required) | Output directory for the `.gputrace` bundle |

Other `ProfilerConfig` fields (`torch_profiler_with_stack`, `torch_profiler_record_shapes`, `delay_iterations`, `max_iterations`, etc.) are torch-profiler-specific and **have no effect** on Metal frame capture in the current implementation. To bound the captured work, control it from the request side (small `max_tokens`, short prompts) rather than via profiler config.

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

That trace is fully usable for static analysis (Summary, Dependencies, kernel ordering, buffer inspection). It is **still too large for Xcode's Profile (replay) button** — see [When frame capture's replay hits a wall](#when-frame-captures-replay-hits-a-wall). For per-kernel timing, use `xctrace` instead (next section).

### Pushing further

If you specifically need an Xcode replay (i.e. you really want the Performance pane populated), you need to push trace size below ~500 MB. Combine all three of:

- `VLLM_METAL_MEMORY_FRACTION=0.05` (or smaller) — further KV shrink
- `--max-model-len=128` — caps the per-request KV allocation independent of `MEMORY_FRACTION`
- `max_tokens=1` from the API — single decode step

Even with all three, expect a few hundred MB. There is no realistic path to making full-engine captures replay-able; that's an Apple-tool scaling limit, not a config-tuning problem.

## Caveats

**Capture is slow.** Apple's frame capture serializes every Metal command (kernel state, buffer contents, pipeline descriptors). For LLM inference — thousands of dispatches per token — wall-clock during capture is **50–100× slower** than uncaptured execution. Plan accordingly: don't capture entire benchmark runs.

**Traces are large.** A single forward pass through Qwen3-0.6B (28 layers) produces a ~2.6 GB `.gputrace` bundle on disk (the logical size, summed across the 30k+ captured buffers, is much higher — Metal uses sparse files). Larger models and longer captures grow accordingly. Make sure your trace dir has space.

**Xcode load time scales with trace size.** Bundles over a few GB take noticeably longer to open. The static panes (Summary, Dependencies, Memory) load fine on multi-GB traces; the **Profile / replay pass does not** — see [When frame capture's replay hits a wall](#when-frame-captures-replay-hits-a-wall). If Xcode beachballs after you click Profile, force-quit and re-run with a smaller `max_tokens` and/or lower `VLLM_METAL_MEMORY_FRACTION`.

**`MTL_CAPTURE_ENABLED` cannot be set lazily.** Apple's framework reads it once at process startup. Our wrapper checks for it in the worker subprocess and raises a clear error if missing. If you forget to set it, you'll see:

```
RuntimeError: Metal frame capture requires MTL_CAPTURE_ENABLED=1 in the
process environment. Restart the engine with that variable set, then retry.
```

**No semantic range markers.** Unlike CUDA's NVTX or torch.profiler's `record_function`, Metal frame capture has no API for inserting named ranges. Kernel-level granularity (each MLX op shows up as a labeled Metal dispatch) is what you get; if you need per-layer or per-stage labels, those would require adding `os_signpost` calls in the C++ extension — not currently supported.

## System-wide Tracing with Instruments

Frame capture only sees the GPU side. For CPU+GPU together (MLX dispatch latency, Python overhead, scheduler timing), use Apple's `xctrace` to record an Instruments trace — no vllm-metal changes needed:

```bash
xctrace record --template "Metal System Trace" --launch -- \
  /usr/bin/env MTL_CAPTURE_ENABLED=0 vllm serve Qwen/Qwen3-0.6B
# Send requests, then Ctrl-C the xctrace command.
# Open the resulting Launch_*.trace bundle in Instruments.app.
```

This produces a `.trace` file (different format from `.gputrace`) that opens in **Instruments.app** (ships with Xcode), giving timeline tracks for both Metal commands and surrounding CPU work.

## Architecture Notes

For contributors:

- The Metal-specific code lives entirely in `vllm_metal/profiler/wrapper.py` (~65 lines). It subclasses `vllm.profiler.wrapper.WorkerProfiler` and implements only `_start` (calls `mx.metal.start_capture`) and `_stop` (calls `mx.metal.stop_capture`).
- `MetalWorker.profile()` in `vllm_metal/v1/worker.py` is the entry point reached by the engine's `collective_rpc("profile", ...)`. It lazily constructs the wrapper on first `start_profile` and reuses it on subsequent starts.
- The state machine for `delay_iterations` / `max_iterations` is inherited from `WorkerProfiler` but **not yet wired** — these fields are no-ops in the current implementation because `MetalWorker` doesn't yet expose `annotate_profile()`, so the engine never calls `WorkerProfiler.step()`. To bound the captured work, callers should use `max_tokens` on `SamplingParams` (offline) or send shorter requests during the start/stop window (server). Wiring `step()` is straightforward future work.
- After editing any source file, run `maturin develop` to relink. Engine workers run in a separate subprocess and load from `site-packages`, so non-editable installs cause stale code to be loaded silently.
