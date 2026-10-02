# Configuration

## Environment Variables

vllm-metal validates these variables when the engine configuration is created and reports every invalid value together, so a typo fails the startup. Boolean switches treat `1` as on and any other value as off.

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_MLX_DEVICE` | `gpu` | MLX device (`gpu` or `cpu`) |
| `VLLM_METAL_DISABLE_NAX` | `0` | Emergency override for automatic M5 NAX prefill attention. Set to `1` to force the non-NAX fallback. |
| `VLLM_METAL_TQ_PREFILL` | `auto` | Materialize eligible TurboQuant prefills when NAX is available. `1` explicitly enables the tiled fallback on other GPUs; `0` keeps compressed attention. Set before worker startup. |
| `VLLM_METAL_TQ_PREFILL_MAX_MIB` | `auto` | TurboQuant prefill workspace, reserved inside `gpu_memory_utilization` before KV sizing. `auto` caps the device allowance by model and scheduler limits for non-speculative serving, and can fall below 256 MiB. A number sets an explicit MiB limit; `0` disables materialization. See [automatic sizing and memory bounds](turboquant.md#prefill-acceleration). |
| `VLLM_METAL_MULTIMODAL_MODE` | `auto` | Multimodal serve mode: `auto` uses the compatibility allowlist (Gemma 4 gets the vision sidecar when its checkpoint allows); `multimodal-native` disables overrides; `text-only` forces the text-only path for every multimodal checkpoint |
| `VLLM_METAL_MM_PREFIX_PATH` | `kernel` | Gemma 4 image-block attention path: `kernel` hands each query row's image-block range to the tiled Metal prefill kernel; `recompute` keeps the MLX SDPA recompute of the block rows after the kernel (the reference path). Any other value fails the startup |
| `VLLM_USE_MODELSCOPE` | `False` | Set True to change model registry to <https://www.modelscope.cn/> |
| `VLLM_METAL_MODELSCOPE_CACHE` | None | Specify the absolute path of the local model |
| `VLLM_METAL_GDN_LAZY_KERNELS` | `1` | Enable lazy GDN kernels for eligible hybrid batches. Set to `0` to force the eager conv / C++ recurrent fallback path. |
| `VLLM_METAL_DECODE_PIPELINE` | `1` | One-step-ahead decode sampling pipeline: eligible pure-decode greedy steps defer the sampling sync one step so the next step's graph build and submit overlap the in-flight GPU forward. Greedy output is unchanged. Disabled automatically when speculative decoding is configured. Set to `0` to force the fully synchronous per-step sample path. |
| `VLLM_METAL_COMPILED_MLP` | `0` | Opt-in compiled stateless-MLP dispatch: decode-shaped MLP/MoE block calls run through an `mx.compile` trace, fusing the per-layer elementwise glue and cutting the per-step op count. Outputs are bitwise identical to the eager dispatch for quantized checkpoints (unquantized fp16 fusion may differ at the ulp level). LoRA serves keep the eager path. Off by default while the dispatch gathers serve mileage; set to `1` to enable. |
| `VLLM_METAL_NATIVE_SAMPLING` | `0` | Opt-in MLX-native non-greedy sampling with temperature/top-k/top-p/min-p for eligible pure-decode batches. Masks are per row, so a batch may mix greedy and random requests with different top-k/top-p/min-p. Seeds, penalties, logprobs, allowed-token IDs, and bad-word constraints use the torch sampler for the whole batch; `min_p` is also supported there. `min_tokens` and `logit_bias` remain unsupported. Sampled tokens use the MLX RNG keyed from the engine seed, so outputs can differ from torch. Set to `1` to enable. |
| `VLLM_METAL_MLA_KERNEL` | `0` | Enable the experimental absorbed-MLA single-pass Metal decode kernel ([RFC #360](https://github.com/vllm-project/vllm-metal/issues/360)). Off by default; the MLA wrapper falls back to the MLX SDPA per-request slow path. Set to `1` to route absorbed-MLA decode through the kernel when the workload matches the instantiated specialization (`kv_lora_rank=512`, `qk_rope_head_dim=64`, `block_size ∈ {16, 32}`, fp16/bf16, decode-only). At startup the engine logs whether decode-only batches take the kernel and, if not, why. |
| `VLLM_METAL_BUILD_FROM_SOURCE` | `0` | Compile the native `_paged_ops` Metal extension from source at runtime instead of loading the prebuilt artifact shipped in the wheel. For kernel developers / source installs; requires the Xcode command-line tools (`clang++`). Off by default — release wheels ship the `.so` prebuilt. See [Contributing](CONTRIBUTING.md). |
| `VLLM_METAL_SPEC_VERIFY_WINDOW` | `0` | Enable spec-decode verification window mode ([issue #465](https://github.com/vllm-project/vllm-metal/issues/465)): the K+1 verification rows share each KV block load instead of re-reading the context per row. Off by default; verify windows keep the expanded per-token layout. Outputs are bitwise identical either way; the win is chip- and shape-dependent (measured up to +40% e2e at concurrency 16-32 with 8k contexts on M2/M3 Ultra, and regressions single-stream on M4 Pro and at concurrency 32 on M2 Max). MLA, hybrid-GDN, and head sizes above 256 always use the expanded layout. The same opt-in also merges the `draft_model` proposer's small committed-token ingest into one window per request (head sizes above 256 keep the expanded layout there too); single-stream TPOT is within run-to-run noise of the expanded ingest (0.6B pair, 8k prefix, M4 Pro), and generated tokens are identical. With speculative decoding on, the engine logs at startup which layout the verify windows and the draft model's committed-token ingest use, and why. |
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

## Multimodal Serve Modes

- `auto`: use the text-only compatibility path only for checkpoints on the compatibility allowlist — Qwen3.5/Qwen3.6 **FP8** conditional-generation wrappers (their `*_weight_scale_inv` tensors are not sanitized by the mlx_vlm loader), and architectures without a multimodal adapter (the Qwen3.6 and MoE wrappers). Non-FP8 Qwen3.5 dense checkpoints (official bf16 and MLX-affine quants) keep the native multimodal path. Gemma 4 checkpoints with vision weights, a loadable HF processor, no per-layer inputs and no speculative decoding serve images through the vision sidecar on the mlx_lm text backbone (see [Supported Models](supported_models.md)); otherwise they stay text-only with a logged reason.
- `multimodal-native`: disable the compatibility fallback and keep the native multimodal path active when validating or developing real multimodal support.
- `text-only`: force the text-only backbone for every multimodal checkpoint, including Gemma 4 (the pre-sidecar behaviour).

The Gemma 4 vision sidecar sets `disable_chunked_mm_input` on the scheduler config so an image block stays inside one prefill step wherever the scheduler allows (the image-block rows need their whole block in the batch). A block that still ends up split falls back to causal attention for that request, with a `falling back to causal attention` warning; see [Supported Models](supported_models.md).

## Speculative Decoding

Pass `--speculative-config` with a JSON object to enable speculative decoding.
Use `--no-async-scheduling` (required for all spec-decode methods on Metal).
See [Speculative Decoding](speculative_decoding.md) for supported methods,
model pairing, and memory considerations.

## KV Cache Memory Settings

The paged KV cache budget follows vLLM's standard `--gpu-memory-utilization`
flag (`gpu_memory_utilization=` for `LLM()`), a fraction in `(0, 1]`. The
former `VLLM_METAL_MEMORY_FRACTION` override has been removed.

Models with full and sliding-window attention use grouped KV cache, allowing
sliding layers to release old blocks. This can improve long-context capacity,
but not necessarily generation speed. Use `--disable-hybrid-kv-cache-manager`
for dense allocation.

The KV pool is allocated lazily. vLLM's allocator zero-fills the backing store,
which on unified memory commits every page of the pool at startup; vllm-metal
requests the same layout without the fill whenever vLLM does not require it
(uniform-precision attention caches — `KVCacheConfig.needs_kv_cache_zeroing`
covers Mamba state and mixed-precision caches, which keep the zero fill). Pages
are then committed only as blocks are used, so `--gpu-memory-utilization` sizes
the *capacity* the engine may reach rather than the resident footprint it pays
for immediately: a 16 GB Mac can give the cache a multi-GB budget while a short
request only occupies the blocks it writes. A slot a request never writes never
reaches an output — the attention kernels mask every position past the sequence
length — so the missing zero fill does not change results.

Unified memory means the pool is not a reservation either way. Nothing pins it:
`mx.set_wired_limit()` covers MLX-allocated buffers, not this torch allocation
that MLX imports in place, so the kernel may compress or swap any of its pages
as soon as something else wants the RAM — zeros from the fill or KV from a
request alike. The fill decides when the pages become resident, not whether the
OS can take them back. Resident KV is bounded by the blocks in use, so a run
that fills the pool peaks at the footprint the eager fill would have committed,
and a run that leaves blocks idle never pays for them. The difference is when
the pages are asked for: the fill paid up front, before the first request, while
a lazy pool faults them in during serving — a long prefill can demand hundreds
of megabytes at once, at the moment the MLX working set is also at its peak.
Faulting a page in under pressure is satisfied from the compressor or swap — it
costs latency, it does not raise an allocation error — and a machine that
exhausts both ends the process the same way it would end a startup zero-fill.
`--gpu-memory-utilization` therefore bounds cache capacity rather than
admission: two processes sized against the same free memory will both claim it,
where the eager fill's resident pages at least showed up as used.

Because of that, the planner checks the plan against the machine before serving:
`VLLM_METAL_KV_COMMIT_PROBE` (on by default) forces a bounded sample of the
planned pool resident at startup — `min(capacity, 512 MiB)`, one write per VM
page — and reads back how much memory was free and how much the kernel had to
push to swap to make room. The sample lives in its own mapping and is dropped as
soon as the probe returns, so it leaves nothing resident behind — the touch
itself is transient, which a process's peak-RSS counter will see but steady
state will not. If the kernel had to page memory out to back the sample, the
machine has no headroom to give and the pool is sized down to what is free, less
a reserve of `max(1 GiB, 1/16 of the recommended working set)`, with a warning.
If free memory is merely below the plan, the pool keeps its capacity and the
shortfall is logged: a plan is a cap, and only the blocks requests actually write
are ever backed. Set `VLLM_METAL_KV_COMMIT_PROBE=0` to skip the touch and trust
`--gpu-memory-utilization` alone.
