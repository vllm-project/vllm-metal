# Configuration

## Environment Variables

vllm-metal validates these variables when the engine configuration is created and reports every invalid value together, so a typo fails the startup. Boolean switches treat `1` as on and any other value as off.

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_MLX_DEVICE` | `gpu` | MLX device (`gpu` or `cpu`) |
| `VLLM_METAL_DISABLE_NAX` | `0` | Emergency override for automatic M5 NAX prefill attention. Set to `1` to force the non-NAX fallback. |
| `VLLM_METAL_NAX_GQA_PREFILL` | `1` | Automatically select three- or six-head grouped NAX prefill on the MLX backend for single-sequence causal attention without sliding windows, softcap or sinks, using BF16, head dimension 256, GQA ratio 6 and 16-token kernel pages. The work-based selector compares query and KV lengths, query/KV heads, detected GPU cores and the actual upstream pipeline. Unsupported features, mixed batches/suffix offsets, unknown capacity or insufficient pipeline limits retain upstream. Extra six-head admission uses complete groups per GPU core and full-tile work coverage for power-of-two KV-head layouts; its guarded-query allowance is calibrated only at 10 cores. Other layouts retain the existing work score; total KV length has no fixed 32K admission cap. Complete tiles omit bounds checks while partial Q/KV tiles remain guarded. Set `0` to retain upstream NAX. Read once at worker startup; restart after changing it. `VLLM_METAL_DISABLE_NAX=1` disables all NAX. |
| `VLLM_METAL_DISABLE_GQA_DECODE` | `0` | The MLX backend automatically uses grouped-query attention (GQA) for eligible one-token decode rows, including the decode prefix of a mixed batch. Eligibility depends on the head layout and the amount of cached context. Set to `1` before starting the server to use the per-token or split-KV kernels for those rows. |
| `VLLM_METAL_TQ_PREFILL` | `auto` | Materialize eligible TurboQuant prefills when NAX is available. `1` explicitly enables the tiled fallback on other GPUs; `0` keeps compressed attention. Set before worker startup. |
| `VLLM_METAL_TQ_PREFILL_MAX_MIB` | `auto` | TurboQuant prefill workspace, reserved inside `gpu_memory_utilization` before KV sizing. `auto` caps the device allowance by model and scheduler limits for non-speculative serving, and can fall below 256 MiB. A number sets an explicit MiB limit; `0` disables materialization. See [automatic sizing and memory bounds](turboquant.md#prefill-acceleration). |
| `VLLM_METAL_MULTIMODAL_MODE` | `auto` | Multimodal serve mode: `auto` uses the compatibility allowlist (Gemma 4 gets the vision sidecar when its checkpoint allows); `multimodal-native` disables overrides; `text-only` forces the text-only path for every multimodal checkpoint |
| `VLLM_METAL_MM_PREFIX_PATH` | `kernel` | Gemma 4 image-block attention path: `kernel` hands each query row's image-block range to the tiled Metal prefill kernel; `recompute` keeps the MLX SDPA recompute of the block rows after the kernel (the reference path). Any other value fails the startup |
| `VLLM_USE_MODELSCOPE` | `False` | Set True to change model registry to <https://www.modelscope.cn/> |
| `VLLM_METAL_MODELSCOPE_CACHE` | None | Specify the absolute path of the local model |
| `VLLM_METAL_GDN_LAZY_KERNELS` | `1` | Enable lazy GDN kernels for eligible hybrid batches. Set to `0` to force the eager conv / C++ recurrent fallback path. |
| `VLLM_METAL_GGUF_FAST_QMV` | `1` | Use SIMD-cooperative native-dtype Metal kernels (`gguf_raw_block.metal`) for GGUF `Q2_K`, `Q3_K`, and `Q6_K` QMV, MoE `gather_matmul`, and dequantization. Set to `0` to use the per-row FP32 reference kernels. |
| `VLLM_METAL_DECODE_PIPELINE` | `1` | One-step-ahead decode sampling pipeline: eligible pure-decode greedy or native-random steps (including seeded requests) defer the sampling sync one step so the next step's graph build and submit overlap the in-flight GPU forward. Greedy output is unchanged. Disabled automatically when speculative decoding is configured. Set to `0` to force the fully synchronous per-step sample path. |
| `VLLM_METAL_COMPILED_MLP` | `0` | Opt-in compiled stateless-MLP dispatch: decode-shaped MLP/MoE block calls run through an `mx.compile` trace, fusing the per-layer elementwise glue and cutting the per-step op count. Outputs are bitwise identical to the eager dispatch for quantized checkpoints (unquantized fp16 fusion may differ at the ulp level). LoRA serves keep the eager path. Off by default while the dispatch gathers serve mileage; set to `1` to enable. |
| `VLLM_METAL_NATIVE_SAMPLING` | `1` | Sample non-greedy requests (temperature, top-k, top-p, min-p) with MLX instead of torch, including prefill first tokens, mixed steps, and deferred decode; greedy output is unchanged, and random outputs can differ from torch for the same seed. Unseeded key derivation is stateless, but the runner keeps a global zero-based counter: random rows share a key hashed from the engine seed and counter in a separate domain from seeded requests. Unseeded draws depend on scheduling, sub-batch order, and batch composition, reproducible only for a fixed request schedule. Each native random sub-batch advances the counter once (decode and prefill separately), while greedy-only, seeded-only, and torch-fallback calls do not. Deferred decode advances it at graph construction, not resolution. Unseeded sequences change from the previous split-chain scheme; seeded request keys are unchanged. A seeded request draws from its own MLX key derived from the seed and its output position, so draws are reproducible given identical logits, sampling parameters, seed, and position. End-to-end token reproducibility also requires a compatible execution environment and batching conditions: batch shape can change logits numerics, and native and torch-fallback paths use different random streams. Seeded deferred decode uses the pre-submit generated-token count as its output position. A sub-batch falls back to torch when any request uses penalties, sample logprobs, allowed-token IDs, bad words, or `logit_bias`, or has not yet generated its `min_tokens`, and every sub-batch does when the step has structured output. With speculation configured, plain decode rows and completed prefills can still use native random sampling. Scheduled drafts use greedy verification. The shared Metal draw changes the random stream, so seeded token sequences can change on upgrade. The PyTorch MPS backend rejects `logit_bias` and `min_tokens`. Set to `0` to sample all non-greedy steps with torch. |
| `VLLM_METAL_MLA_KERNEL` | `0` | Enable the experimental absorbed-MLA single-pass Metal decode kernel ([RFC #360](https://github.com/vllm-project/vllm-metal/issues/360)). Off by default; the MLA wrapper falls back to the MLX SDPA per-request slow path. Set to `1` to route absorbed-MLA decode through the kernel when the workload matches the instantiated specialization (`kv_lora_rank=512`, `qk_rope_head_dim=64`, `block_size ∈ {16, 32}`, fp16/bf16, decode-only). At startup the engine logs whether decode-only batches take the kernel and, if not, why. |
| `VLLM_METAL_BUILD_FROM_SOURCE` | `0` | Compile the native `_paged_ops` Metal extension from source at runtime instead of loading the prebuilt artifact shipped in the wheel. For kernel developers / source installs; requires the Xcode command-line tools (`clang++`). Off by default — release wheels ship the `.so` prebuilt. See [Contributing](CONTRIBUTING.md). |
| `VLLM_METAL_SPEC_VERIFY_WINDOW` | `0` | Enable spec-decode verification window mode ([issue #465](https://github.com/vllm-project/vllm-metal/issues/465)): the K+1 verification rows share each KV block load instead of re-reading the context per row. Off by default; non-hybrid verify windows keep the expanded per-token layout. Outputs are bitwise identical either way; the win is chip- and shape-dependent (measured up to +40% e2e at concurrency 16-32 with 8k contexts on M2/M3 Ultra, and regressions single-stream on M4 Pro and at concurrency 32 on M2 Max). Non-hybrid MLA models and head sizes above 256 use the expanded layout. Hybrid GDN targets force window mode when speculative decoding is configured (their state scan needs one segment per request); hybrid speculation with head sizes above 256 or non-GDN state families is rejected. The same opt-in also merges the `draft_model` proposer's small committed-token ingest into one window per request (head sizes above 256 keep the expanded layout there too); single-stream TPOT is within run-to-run noise of the expanded ingest (0.6B pair, 8k prefix, M4 Pro), and generated tokens are identical. With speculative decoding on, the engine logs at startup which layout the verify windows and the draft model's committed-token ingest use, and why. |
| `VLLM_METAL_SPEC_INGEST_CHUNK` | `1024` | Maximum number of cold draft-model KV-ingest tokens processed per forward ([issue #482](https://github.com/vllm-project/vllm-metal/issues/482)). Chunking bounds each dispatch stall and the peak logits allocation; a multiple of the KV block size is recommended. Set to `0` to restore single-forward ingest; a non-integer or negative value fails the startup. |
| `VLLM_METAL_VISIBLE_DEVICES` | — | Set automatically by the Ray executor per worker (the device-control var); not user-configurable. See [Distributed](distributed.md). |
| `VLLM_METAL_RING_BASE_PORT` | `32323` | Base TCP port for the MLX ring data plane under pipeline parallelism, in the user-port range `[1024, 65535]`; stage *r* binds `base + r` (so the default is `32323`/`32324` for two stages). Set the **same** value on every node to move the ring off a busy port — e.g. when an `mlx.launch` job, a restart still in `TIME_WAIT`, or another PP job holds the default. See [Distributed](distributed.md#pipeline-parallelism). |

## MLX Command-Buffer Defaults

On macOS the plugin defaults `MLX_MAX_OPS_PER_BUFFER` to `2000` via
`setdefault`, so a value you export yourself always wins. MLX's own default is
sized for small generate loops; a vLLM decode step on a large MoE model builds
thousands of lazy ops per step, and the resulting per-buffer commit overhead
slows the step submit. `2000` sits on the measured plateau.

`MLX_MAX_MB_PER_BUFFER` trades transient profile memory for per-step commit
overhead. The plugin defaults it to `2000` when the usable budget (total
memory times the effective memory fraction) is at least 90 GiB, and leaves it
unset below that, on Ray executors, or when `max_num_batched_tokens` exceeds
4096 (the #585 startup-failure shape). A value you export yourself always
wins. Outputs are unaffected.

## PyTorch MPS Allocator Watermark

On the PyTorch MPS backend each worker process defaults
`PYTORCH_MPS_LOW_WATERMARK_RATIO` to its effective memory fraction
(`--gpu-memory-utilization`), so PyTorch starts returning cached heaps once
the pool passes the configured budget. Colocated Omni stages each use their
own stage's fraction. PyTorch's own default is 1.4 times the recommended
working set, which an 8 GB Mac cannot reach before Metal runs out of memory.
Exporting either `PYTORCH_MPS_LOW_WATERMARK_RATIO` or
`PYTORCH_MPS_HIGH_WATERMARK_RATIO` yourself leaves both untouched.

## Multimodal Serve Modes

- `auto`: use the native multimodal path where it is supported. Qwen3.5/Qwen3.6
  **FP8** wrappers use the text backbone because mlx-vlm does not sanitize their
  `*_weight_scale_inv` tensors. Architectures without a matching adapter,
  compatible Mistral 3, and Muse Glimmer also use the text backbone. Non-FP8
  Qwen3.5 dense and MoE architectures, including those used by Qwen3.6
  checkpoints, keep the native path. GGUF checkpoints use the text-only path
  because they contain no vision tower. Gemma 4 checkpoints serve images
  through the vision sidecar when they have vision weights, a loadable HF
  processor, and no speculative decoding; otherwise they stay text-only with a
  logged reason. See [Supported Models](supported_models.md).
- `multimodal-native`: disable the compatibility fallback and keep the native multimodal path active when validating or developing real multimodal support.
- `text-only`: force the text-only backbone for every multimodal checkpoint, including Gemma 4 (the pre-sidecar behaviour).

The Gemma 4 vision sidecar sets `disable_chunked_mm_input` on the scheduler config so an image block stays inside one prefill step wherever the scheduler allows (the image-block rows need their whole block in the batch). A block that still ends up split falls back to causal attention for that request, with a `falling back to causal attention` warning; see [Supported Models](supported_models.md).

## Speculative Decoding

Pass `--speculative-config` with a JSON object to enable speculative decoding.
Use `--no-async-scheduling` (required for all spec-decode methods on Metal).
See [Speculative Decoding](speculative_decoding.md) for supported methods,
model pairing, and memory considerations.

## Sleep Mode

Start with `--enable-sleep-mode` (or `LLM(..., enable_sleep_mode=True)`) to
give the KV cache's memory back while the engine is idle, for example when a
model server stays open in the background on a laptop, or when the same Mac
runs another memory-hungry job between requests. The API is vLLM's
[sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode.html); the
`/sleep`, `/wake_up` and `/is_sleeping` endpoints and `LLM.sleep()` work the
same way.

Sleep levels mean something different on unified memory than on a GPU:

| Level | vLLM on a GPU | vllm-metal |
|---|---|---|
| 1 | Copies weights to CPU memory, discards the KV cache | Discards the KV cache. Weights stay in place: unified memory has nowhere cheaper to copy them to. |
| 2 | Discards weights and the KV cache | Not supported yet. The call fails after vLLM has paused the engine and, with the default `mode="abort"`, aborted running requests. `wake_up()` resumes the engine. |

- `llm.sleep(level=1)` frees the KV cache. `llm.wake_up()` re-creates it in
  well under a second, and the prefix cache starts empty.
- `llm.sleep(level=0)` followed by `llm.release_kv_cache_memory()` also frees
  the KV cache. Restore it with `llm.wake_up(tags=["kv_cache"])`.
- Either way the whole KV pool is freed. The pool takes that memory again at
  wake or with the first request after wake.
- With [KV offloading](kv_offloading.md), sleep also frees the host pool. It
  comes back empty at wake, even with `reset_connector=False`. This differs
  from vLLM on a GPU, where `False` keeps the offloaded blocks for requests
  after wake. On a Mac the host pool sits in the unified memory that sleep
  gives back. Prompts already on the disk tier still restore.

Sleep mode supports dense, hybrid and MLA models on the MLX backend.
Supported speculative decoding methods are `ngram`, `draft_model`, `dflash` and
`dspark`.

## KV Cache Memory Settings

The paged KV cache budget follows vLLM's standard `--gpu-memory-utilization`
flag (`gpu_memory_utilization=` for `LLM()`), a fraction in `(0, 1]`. The
former `VLLM_METAL_MEMORY_FRACTION` override has been removed.

Set `--kv-cache-memory-bytes` (`kv_cache_memory_bytes=` for `LLM()`) to a
positive byte count to **override** automatic sizing, for example
`--kv-cache-memory-bytes 2147483648` for 2 GiB. This budgets paged cache storage;
weights, activations, runtime workspace, and the
[offload host pool](kv_offloading.md) need additional memory.

Sizes round down to whole blocks. Startup profiling and the memory probe still
run; an explicit budget that would need shrinking is rejected. The separate
`--num-gpu-blocks-override` takes precedence over byte-derived block counts.

Models with full and sliding-window attention use grouped KV cache, allowing
sliding layers to release old blocks. This can improve long-context capacity,
but not necessarily generation speed. Use `--disable-hybrid-kv-cache-manager`
for dense allocation.

Plan for the KV pool as RAM the server holds at its full size, even when
requests use little of it. Expect the whole pool in memory at startup or after
the first request
([#1198](https://github.com/vllm-project/vllm-metal/issues/1198)). The memory
reported for the process (RSS) can show less than this.

With automatic sizing, the pool and the model weights together take about the
`--gpu-memory-utilization` fraction (default 0.92) of the memory macOS
recommends for GPU use (Metal's recommended working set). To leave room for
other applications, lower the fraction or set a smaller pool with
`--kv-cache-memory-bytes`. Each server sizes itself as if it were alone, so two
servers at the same fraction each plan for that full fraction.

By default, vllm-metal tests a sample of the planned pool, up to 512 MiB,
against the machine at startup:

- If macOS has to compress or swap memory to hold the sample, the pool can be
  sized down, with a warning. Where a pool set with `--kv-cache-memory-bytes`
  would be sized down, startup stops with an error instead.
- If the log says the pool is larger than this machine's free memory, the pool
  keeps its size. Make the pool smaller or close other applications.
- If memory statistics cannot be read, startup stops with an error.

The test covers a sample, so leave headroom even when no warning appears. Set
`VLLM_METAL_KV_COMMIT_PROBE=0` to skip the test and keep the planned pool
size.
