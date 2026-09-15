# Configuration

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_MLX_DEVICE` | `gpu` | MLX device (`gpu` or `cpu`) |
| `VLLM_METAL_DISABLE_NAX` | `0` | Emergency override for automatic M5 NAX prefill attention. Set to `1` to force the non-NAX fallback. |
| `VLLM_METAL_MULTIMODAL_MODE` | `auto` | Multimodal serve mode: `auto` uses the compatibility allowlist; `multimodal-native` disables overrides |
| `VLLM_USE_MODELSCOPE` | `False` | Set True to change model registry to <https://www.modelscope.cn/> |
| `VLLM_METAL_MODELSCOPE_CACHE` | None | Specify the absolute path of the local model |
| `VLLM_METAL_GDN_LAZY_KERNELS` | `1` | Enable lazy GDN kernels for eligible hybrid batches. Set to `0` to force the eager conv / C++ recurrent fallback path. |
| `VLLM_METAL_DECODE_PIPELINE` | `1` | One-step-ahead decode sampling pipeline: eligible pure-decode greedy steps defer the sampling sync one step so the next step's graph build and submit overlap the in-flight GPU forward. Greedy output is unchanged. Disabled automatically when speculative decoding is configured. Set to `0` to force the fully synchronous per-step sample path. |
| `VLLM_METAL_COMPILED_MLP` | `0` | Opt-in compiled stateless-MLP dispatch: decode-shaped MLP/MoE block calls run through an `mx.compile` trace, fusing the per-layer elementwise glue and cutting the per-step op count. Outputs are bitwise identical to the eager dispatch for quantized checkpoints (unquantized fp16 fusion may differ at the ulp level). LoRA serves keep the eager path. Off by default while the dispatch gathers serve mileage; set to `1` to enable. |
| `VLLM_METAL_NATIVE_SAMPLING` | `0` | Opt-in MLX-native non-greedy sampling with temperature/top-k/top-p/min-p for eligible pure-decode batches. Native batches must share top-k, top-p, and min-p. Seeds, penalties, logprobs, allowed-token IDs, and bad-word constraints use the torch sampler; `min_p` is also supported there. `min_tokens` and `logit_bias` remain unsupported. Sampled tokens use the MLX RNG keyed from the engine seed, so outputs can differ from torch. Set to `1` to enable. |
| `VLLM_METAL_MLA_KERNEL` | `0` | Enable the experimental absorbed-MLA single-pass Metal decode kernel ([RFC #360](https://github.com/vllm-project/vllm-metal/issues/360)). Off by default; the MLA wrapper falls back to the MLX SDPA per-request slow path. Set to `1` to route absorbed-MLA decode through the kernel when the workload matches the instantiated specialization (`kv_lora_rank=512`, `qk_rope_head_dim=64`, `block_size ∈ {16, 32}`, fp16/bf16, decode-only). |
| `VLLM_METAL_BUILD_FROM_SOURCE` | `0` | Compile the native `_paged_ops` Metal extension from source at runtime instead of loading the prebuilt artifact shipped in the wheel. For kernel developers / source installs; requires the Xcode command-line tools (`clang++`). Off by default — release wheels ship the `.so` prebuilt. See [Contributing](CONTRIBUTING.md). |
| `VLLM_METAL_SPEC_VERIFY_WINDOW` | `0` | Enable spec-decode verification window mode ([issue #465](https://github.com/vllm-project/vllm-metal/issues/465)): the K+1 verification rows share each KV block load instead of re-reading the context per row. Off by default; verify windows keep the expanded per-token layout. Outputs are bitwise identical either way; the win is chip- and shape-dependent (measured up to +40% e2e at concurrency 16-32 with 8k contexts on M2/M3 Ultra, and regressions single-stream on M4 Pro and at concurrency 32 on M2 Max). MLA, hybrid-GDN, and head sizes above 256 always use the expanded layout. The same opt-in also merges the `draft_model` proposer's small committed-token ingest into one window per request (head sizes above 256 keep the expanded layout there too); single-stream TPOT is within run-to-run noise of the expanded ingest (0.6B pair, 8k prefix, M4 Pro), and generated tokens are identical. |
| `VLLM_METAL_SPEC_INGEST_CHUNK` | `1024` | Maximum number of cold draft-model KV-ingest tokens processed per forward ([issue #482](https://github.com/vllm-project/vllm-metal/issues/482)). Chunking bounds each dispatch stall and the peak logits allocation; a multiple of the KV block size is recommended. Set to `0` to restore single-forward ingest. |
| `VLLM_METAL_VISIBLE_DEVICES` | — | Set automatically by the Ray executor per worker (the device-control var); not user-configurable. See [Distributed](distributed.md). |
| `VLLM_METAL_RING_BASE_PORT` | `32323` | Base TCP port for the MLX ring data plane under pipeline parallelism; stage *r* binds `base + r` (so the default is `32323`/`32324` for two stages). Set the **same** value on every node to move the ring off a busy port — e.g. when an `mlx.launch` job, a restart still in `TIME_WAIT`, or another PP job holds the default. See [Distributed](distributed.md#pipeline-parallelism). |

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

- `auto`: use the text-only compatibility path for checkpoints on the compatibility allowlist, such as Gemma4 and Qwen3.5/Qwen3.6 FP8 conditional-generation wrappers.
- `multimodal-native`: disable the compatibility fallback and keep the native multimodal path active when validating or developing real multimodal support.

## Speculative Decoding

Pass `--speculative-config` with a JSON object to enable speculative decoding.
Use `--no-async-scheduling` (required for all spec-decode methods on Metal).
See [Speculative Decoding](speculative_decoding.md) for supported methods,
model pairing, and memory considerations.

## KV Cache Memory Settings

The paged KV cache budget follows vLLM's standard `--gpu-memory-utilization`
flag (`gpu_memory_utilization=` for `LLM()`), a fraction in `(0, 1]`. The
former `VLLM_METAL_MEMORY_FRACTION` override has been removed.

## Bounded GDN State Cache

Hybrid GDN models can opt into a separately bounded state pool:

```sh
vllm serve /path/to/model \
  --enable-prefix-caching --gpu-memory-utilization 0.8 --max-num-seqs 4 \
  --additional-config '{"state_cache_budget_mib": 2048}'
```

The `2048` value is an example. Size it from the row formula above, the target
concurrency, whether asynchronous scheduling is enabled, how many idle
checkpoints to keep, and the host's planned Metal budget. The published 27B
pairs use 2048 MiB so that those archives stay comparable; they are not a
recommendation to use 2048 MiB on every machine.

`state_cache_budget_mib` is a positive integer measured in MiB (2^20 bytes).
Omit the key to keep the existing compact, demand-grown state cache. This is
an experimental, per-engine option; it does not change the model weights or
the attention/KV page geometry.

The option limits **stable GDN state arrays**, including working states and
retained prefix checkpoints. Forward activations, temporary row copies and
pending compact state updates are outside this limit. The planner preserves
the existing runtime-overhead estimate and adds a separate allowance for
batched state copies, compact updates and scatter staging. The single-sequence
startup profile does not prove an upper bound for every concurrent workload.
Measure peak memory with the intended batch sizes. This setting is not a
process-wide memory limit or an operating-system RAM reservation. The planner
deducts the fixed state allocation and the additional allowance once, then gives
the remaining planned memory to KV blocks, instead of reserving GDN state for
every possible block ID. Both scheduler and worker derive and check the same
physical-row budget from the cache layout.

The state pool is allocated once, **after** shared physical pools are resolved.
This eliminates state-pool growth copies and their overlap reservation. Unlike
the default demand-grown mode, the selected bounded pool is fully allocated
even at low load. Additional KV capacity also consumes real memory at startup.

When a new state would exceed the quota, the scheduler first retires uncached
idle states, then evicts unreferenced checkpoints in the BlockPool's free-list
order. Eviction removes every prefix-hash alias. Active requests, pending GPU
work and the checkpoint selected for an incoming restore remain protected.
Worker steps carry ordered block generations, so an old global block ID cannot
silently restore the state of its next owner. State-slot reuse settles pending
state writes and waits for the GPU. The independent pending token buffer keeps
its normal decode-pipeline delivery order and sampling path.

Each running request reserves room for its completed source plus one new state
per possible in-flight batch, across all GDN groups. The effective running
limit is the smaller of `max_num_seqs` and the number that this state budget
can support. The startup log reports that limit. A smaller budget may therefore
reduce concurrency or checkpoint hit rate, even while it increases KV capacity.
The scheduler queues requests that cannot yet be admitted; the worker never
grows beyond the state limit to satisfy them. A budget too small for one request
to make progress is rejected.

A resumable streaming-input request keeps its admission reservation while it
owns a state, including between input chunks. When its next chunk arrives, the
scheduler resumes that existing owner before admitting new requests. Aborting
or completing the request releases ownership; in-flight physical rows still
wait for their GPU fence. Ordinary streaming output uses the same request
lifecycle as non-streaming output.

Do not copy one MiB value across machines or serving shapes. Omitting
`state_cache_budget_mib` keeps the compact, demand-grown state cache. When the
option is set, the working reserve is `(B + 1) * G` physical rows per running
request, where `B` is `max_concurrent_batches` (1 when asynchronous scheduling
is off, otherwise the engine's in-flight batch depth) and `G` is the number of
striped GDN groups. The scheduler admits at most `floor(capacity / ((B + 1) *
G))` running requests. A budget smaller than one request's working rows is
rejected at startup. Extra rows above that floor are available for idle
prefix checkpoints; they are not required for a single in-flight request.

For Qwen3.8-27B with the current BF16-conv/FP32-recurrent geometry, a physical
row across 16 shared pools is 48.9375 MiB and `G = 3`. One synchronous request
therefore needs 6 rows (294 MiB). Four asynchronous requests at `B = 2` need 36
working rows (1762 MiB) before any spare checkpoint capacity. The 2048 MiB
serve example above admits 41 rows, which is a reproducible **benchmark** setting for
that four-request asynchronous 27B matrix on 48 GB / 64 GB hosts, not a default
for every RAM size. On a tighter host, a one-request 294 MiB budget can assign
more memory to KV than 2048 MiB, because the stable pool is fully allocated at
startup. Other models and dtypes have different row sizes; use the resolved
slot size and running-request limit printed at startup.

The additional state scratch allowance is `(3 * B + 1) * R * F`, where `B` is
the maximum number of in-flight batches, `R` is the admitted request limit
capped by the token budget per batch, and `F` is one request's state across all
logical GDN layers. Three payloads per batch cover copy/zero scratch, new
compact updates, and possible contiguous/dtype staging; one more covers
preceding pending updates. This conservative estimate deliberately counts some
aliases and non-overlapping work separately, without subtracting presumed
overlap with the startup profile. It is an allowance for known state-shaped
temporaries, not a bound on all Metal/allocator/activation memory. In the 27B
example with four admitted requests and two in-flight batches, it reserves
another 4110.75 MiB, in addition to the 2006.4375 MiB stable pool and the ordinary
profiled overhead. Both allowances are printed before KV allocation.

The first version supports local GDN **align** serving with FCFS scheduling,
full-block prefix hits, and either synchronous or asynchronous scheduling.
It rejects speculative decoding, priority scheduling, custom prefix-match
units, non-local executors, tensor/pipeline/data parallelism, KV offloading or
transfer connectors, and non-GDN state families. These combinations need
additional lifecycle validation; they are not silently disabled. The scheduler
adapter depends on the vLLM core interfaces tested with this checkout's pinned
release.

See [State-cache budget benchmark](state-cache-budget-benchmark.md) for paired
baseline/budget runs, full-output comparison, and separate reporting of actual
memory, configured capacity and observed workload limits.
