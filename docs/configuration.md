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
| `VLLM_METAL_DECODE_PIPELINE` | `1` | One-step-ahead decode sampling pipeline: eligible pure-decode greedy steps defer the sampling sync one step so the next step's graph build and submit overlap the in-flight GPU forward. Greedy output is unchanged. With speculative decoding configured the pipeline runs only on steps a DSpark proposer will not draft at (the `bypass` mode, or the `adaptive` planner's bypass decision for the batch; the drafter keeps ingesting target features on such steps) and stays off for the other proposers. Set to `0` to force the fully synchronous per-step sample path. |
| `VLLM_METAL_COMPILED_MLP` | `0` | Opt-in compiled stateless-MLP dispatch: decode-shaped MLP/MoE block calls run through an `mx.compile` trace, fusing the per-layer elementwise glue and cutting the per-step op count. Outputs are bitwise identical to the eager dispatch for quantized checkpoints (unquantized fp16 fusion may differ at the ulp level). LoRA serves keep the eager path. Off by default while the dispatch gathers serve mileage; set to `1` to enable. |
| `VLLM_METAL_NATIVE_SAMPLING` | `0` | Opt-in MLX-native non-greedy sampling with temperature/top-k/top-p/min-p for eligible pure-decode batches. Native batches must share top-k, top-p, and min-p. Seeds, penalties, logprobs, allowed-token IDs, and bad-word constraints use the torch sampler; `min_p` is also supported there. `min_tokens` and `logit_bias` remain unsupported. Sampled tokens use the MLX RNG keyed from the engine seed, so outputs can differ from torch. Set to `1` to enable. |
| `VLLM_METAL_MLA_KERNEL` | `0` | Enable the experimental absorbed-MLA single-pass Metal decode kernel ([RFC #360](https://github.com/vllm-project/vllm-metal/issues/360)). Off by default; the MLA wrapper falls back to the MLX SDPA per-request slow path. Set to `1` to route absorbed-MLA decode through the kernel when the workload matches the instantiated specialization (`kv_lora_rank=512`, `qk_rope_head_dim=64`, `block_size ∈ {16, 32}`, fp16/bf16, decode-only). |
| `VLLM_METAL_BUILD_FROM_SOURCE` | `0` | Compile the native `_paged_ops` Metal extension from source at runtime instead of loading the prebuilt artifact shipped in the wheel. For kernel developers / source installs; requires the Xcode command-line tools (`clang++`). Off by default — release wheels ship the `.so` prebuilt. See [Contributing](CONTRIBUTING.md). |
| `VLLM_METAL_SPEC_VERIFY_WINDOW` | `0` | Enable spec-decode verification window mode ([issue #465](https://github.com/vllm-project/vllm-metal/issues/465)): the K+1 verification rows share each KV block load instead of re-reading the context per row. Off by default; verify windows keep the expanded per-token layout. Outputs are bitwise identical either way; the win is chip- and shape-dependent (measured up to +40% e2e at concurrency 16-32 with 8k contexts on M2/M3 Ultra, and regressions single-stream on M4 Pro and at concurrency 32 on M2 Max). MLA, hybrid-GDN, and head sizes above 256 always use the expanded layout. The same opt-in also merges the `draft_model` proposer's small committed-token ingest into one window per request (head sizes above 256 keep the expanded layout there too); single-stream TPOT is within run-to-run noise of the expanded ingest (0.6B pair, 8k prefix, M4 Pro), and generated tokens are identical. |
| `VLLM_METAL_DSPARK_MAX_CONTEXTS` | `32` | Upper bound on concurrent DSpark draft contexts. The planner reserves `min(--max-num-seqs, value)` complete contexts before target KV allocation (20 KiB per context token per request for the Qwen3-4B drafter), so a larger value costs target KV capacity. A request scheduled while every slot is held uses target-only generation for its lifetime, because a context needs the target features of every earlier position; slots are reused as requests finish. |
| `VLLM_METAL_DSPARK_MODE` | `fixed` | DSpark serving mode. `fixed` drafts and verifies the configured `num_speculative_tokens` for every eligible request. `adaptive` plans each request's draft prefix from its calibrated confidence and the measured step costs, and decides before running the drafter whether drafting the batch pays at all (bypass reasons are counted); it needs both artifacts below and fails at startup with the reason otherwise. `bypass` keeps the drafter loaded and every context advancing but never drafts: the step the adaptive planner weighs drafting against, for profiling and A/B serving. |
| `VLLM_METAL_DSPARK_LAPSE` | `1` | Load regime of the adaptive and bypass modes. When the planner would decline to draft a batch of the step's size on 32 consecutive steps, the proposer lapses: it stops capturing and ingesting target features and releases every draft context, so a server that cannot profit from drafting at the current load costs what target-only serving costs; it primes new requests again once the load has dropped below the count it lapsed at and the planner would draft, on 8 consecutive steps (requests that ran through a lapse keep target-only generation). The bypass mode lapses from the start. Set to `0` to keep every context current at all loads. |
| `VLLM_METAL_DSPARK_DRAFT_PRECISION` | `quantized` | DSpark drafter weights at load: `quantized` converts the drafter to the target's affine 4-bit recipe (the qualified default); `source` keeps the checkpoint's own precision (bfloat16 for the released drafters), which costs more memory and changes the draft cost profile; use it for acceptance and speed comparisons and re-profile the cost model for it. |
| `VLLM_METAL_DSPARK_CALIBRATION` | — | Path to the confidence calibration artifact (`tools/dspark_confidence_calibrate.py fit`) for the served pair; validated against the pair's manifest. Required by the adaptive mode. |
| `VLLM_METAL_DSPARK_COST_MODEL` | — | Path to the measured cost model (`tools/dspark_cost_profile.py`: step costs through the serving path over request counts, drafted widths and decode contexts) of the served pair on this machine; validated against the pair's manifest. Required by the adaptive mode. |
| `VLLM_METAL_DSPARK_MAX_DRAFTS_PER_STEP` | `0` | Maximum requests drafted per scheduler step, bounding the verification rows one step adds. `0` drafts every eligible request. When the cap binds, requests rotate least-recently-drafted first, so none starves; contexts of waiting requests still advance every step. |
| `VLLM_METAL_DSPARK_PAGED_CONTEXT` | `0` | Keep the drafter's context K/V in a paged pool in the target's own block layout (block size 16) instead of a private per-request arena, read by the same paged-attention kernel the target uses. The pool holds the same worst case but hands pages out as contexts grow, and needs no reservation for a whole-buffer rewrite, which returns that memory to the target's KV cache. Requires a drafter head dimension the kernels are built for (64, 80, 96, 112, 128, 192, 256, 512); any other keeps the arena and logs a warning. |
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
