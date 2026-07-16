# KV offloading connector for the Metal backend — design

Status: design accepted, implementation in progress (branch `kv-offload-metal`).
Target: vLLM 0.25.1's `OffloadingConnector` (`--kv-offloading-backend native
--kv-offloading-size N`) working on Apple Silicon, upstreamable to
vllm-project/vllm-metal.

## 1. Why offloading makes sense on unified memory

Apple Silicon has one DRAM pool, so "GPU→CPU offload" sounds like a no-op. It
isn't, because vllm-metal's KV pool is **wired** (non-pageable): the worker
calls `mx.set_wired_limit` (`vllm_metal/utils.py:56`) and the paged cache is
budgeted against `max_recommended_working_set_size × VLLM_METAL_MEMORY_FRACTION`
(`vllm_metal/v1/cache_policy.py:680-709`). The tiers on this platform are:

1. **Wired MLX KV pool** — fast, protected, hard-capped.
2. **Pageable host memory** (plain `numpy` allocations) — same DRAM, but macOS
   can reclaim it under pressure; it extends KV capacity beyond the wired cap.
3. **NVMe disk** (via vLLM's tiering/`fs` secondary tier) — capacity beyond RAM
   and persistence; Apple NVMe at 5–8 GB/s beats long-prefix recompute.

A note on vocabulary: upstream names the primary offload target the "CPU
tier" (`CPUOffloadingSpec`, `CPULoadStoreSpec`, `medium() == "CPU"`). That
device-vs-host framing is wrong for any shared-memory architecture — Apple
Silicon here, and equally unified-memory machines like NVIDIA DGX Spark
(Grace-class): there is no second memory, only the wired/pageable boundary
above. This port keeps upstream's names at the code level to stay a
minimal-surface change; prose in this document says "host pool" and means
tier 2. A restore from the host pool saves prefill *compute*, not memory-bus
distance — that is the entire economics of offloading on this platform.

## 2. What vLLM already provides (all reusable, verified on 0.25.1)

The connector splits into a scheduler half and a worker half; both are
instantiated from the same class by role
(`vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:46`).

Platform-agnostic and reused as-is:

- **Scheduler half** — `OffloadingConnectorScheduler`: lookup/hit accounting,
  load/store job construction, `jobs_to_flush` fencing. Needs only
  `spec.get_manager()`.
- **`CPUOffloadingManager`** (`vllm/v1/kv_offload/cpu/manager.py`) — block
  pool, ref-counts, LRU/ARC policies, store-threshold filter. Pure Python over
  block ids and hash bytes.
- **`OffloadingWorker`** (`vllm/v1/kv_offload/base.py`) — the abstract worker
  contract (`submit_store`/`submit_load`/`get_finished`/`wait`); the connector
  worker holds one worker directly and submits jobs to it — no
  `(src, dst)`-medium routing.
- **`OffloadingConnectorWorker`** — job bookkeeping, deferred stores,
  `finished_recving` reporting, bandwidth stats.
- **Spec/factory plumbing** — `OffloadingSpecFactory` resolves
  `kv_connector_extra_config["spec_name"]` (default `CPUOffloadingSpec`) with a
  `spec_module_path` dynamic-import fallback (`vllm/v1/kv_offload/factory.py:37-45`)
  — an extension hook that requires **zero vLLM edits**.
- **CLI translation** — `--kv-offloading-size N` + backend `native` →
  `kv_connector="OffloadingConnector"`, `kv_role="kv_both"`,
  `extra_config["cpu_bytes_to_use"] = N GiB` (`vllm/config/vllm.py:785-799`).
- **Tiering** — `TieringOffloadingSpec` + `FileSystemTierManager`
  (`vllm/v1/kv_offload/tiering/fs/manager.py`) give a disk tier whose I/O path
  is platform-agnostic (`O_DIRECT` degrades to buffered on macOS). M2 target.

The **only CUDA-bound pieces** are:

- `CPUOffloadingSpec.get_worker` hard-raises unless
  `is_cuda_alike() or is_xpu()` (`vllm/v1/kv_offload/cpu/spec.py:146-150`).
- The copy engine `vllm/v1/kv_offload/cpu/gpu_worker.py` — CUDA streams,
  events, `cudaHostRegister`, `swap_blocks_batch` (`cuMemcpyBatchAsync`),
  Triton UVA kernel.
- `SharedOffloadRegion` — Linux `/dev/shm` mmap (no macOS equivalent needed:
  single-process, plain host allocations suffice).
- Worker-side canonicalization
  (`.../kv_connector/v1/offloading/worker.py:50-222`): reinterprets each
  layer's **torch** KV tensor as an int8 `(num_blocks, page_size_bytes)`
  storage view.

## 3. Why the torch canonicalization cannot be reused on Metal

vllm-metal's paged KV cache is **per-layer MLX arrays**
(`vllm_metal/attention/caches/kv_cache.py:34`):

- `key_caches[L]` and `value_caches[L]` are **separate** arrays of shape
  `(num_blocks, block_size, kv_heads, head_dim)` (fp16/bf16/fp32); TurboQuant
  adds packed K/V plus three fp16 scale/zero arrays per layer.
- vLLM's canonical contract wants **one** tensor per layer where one block =
  `page_size_bytes` **contiguous** bytes covering K and V. Two separate arrays
  can never satisfy that; a layout change to interleave K/V per block would
  invalidate the Metal attention kernels.
- The KV write kernels (`reshape_and_cache`, `tq_encode`,
  `vllm_metal/attention/impls/sdpa.py:524-557`) mutate the underlying Metal
  buffers **in place** but return *new* `mx.array` objects (fresh lazy-graph
  provenance) that are rebound onto `kv_cache.key_caches[L]` every layer call.
  A torch tensor aliasing the buffer would (a) not see graph provenance, so
  reads race pending writes, and (b) pin an extra reference that defeats MLX
  donation. Aliasing is out.

Therefore: **keep vLLM's scheduler half, manager, and worker orchestration;
replace only the registration entry point and the copy engine with
MLX-native code.** All block-movement decisions stay upstream; vllm-metal
contributes a data mover.

## 4. Architecture

New package `vllm_metal/v1/kv_offload/` with three pieces, plus small hooks in
platform/worker/model-runner:

```
vLLM scheduler process                 Metal worker (same proc under "uni" executor)
──────────────────────                 ─────────────────────────────────────────────
OffloadingConnector(SCHEDULER)         MetalOffloadingConnector(WORKER)  [subclass]
 └─ OffloadingConnectorScheduler        ├─ OffloadingConnectorWorker (reused)
     └─ CPUOffloadingManager (reused)   │    └─ MetalKVOffloadWorker         [new]
                                        │       (implements OffloadingWorker)
                                        └─ MetalOffloadingSpec               [new]
                                             ├─ get_manager() → CPUOffloadingManager
                                             └─ host pool: per-layer numpy arrays
```

### 4.1 `MetalOffloadingSpec(CPUOffloadingSpec)`

- Inherits all sizing: `num_blocks` (CPU-side) from `cpu_bytes_to_use` ÷
  bytes-per-offloaded-block, computed from `kv_cache_config.kv_cache_tensors`
  (`cpu/spec.py:60-104`) — vllm-metal populates these sizes, including the
  TurboQuant packed layout (see `cache_policy.py`, PR #472).
- Inherits `get_manager()` unchanged.
- Overrides `get_metal_worker(...)` to build the Metal worker from the live
  `MetalPagedKVCache` instead of `CanonicalKVCaches` — no CUDA guard.
- Selected via `kv_connector_extra_config["spec_name"] = "MetalOffloadingSpec"`
  + `spec_module_path`, injected by `MetalPlatform.check_and_update_config`.

### 4.2 `MetalOffloadingConnector(OffloadingConnector)`

Registered with `KVConnectorFactory.register_connector(...)` at plugin init;
`MetalPlatform.check_and_update_config` rewrites
`kv_transfer_config.kv_connector` from `"OffloadingConnector"` to
`"MetalOffloadingConnector"` so the user-facing CLI is identical to CUDA.

Overrides exactly one worker-side method:

- `register_kv_caches(kv_caches)` — accepts the `MetalPagedKVCache` (passed by
  our model runner, which owns the call site) and sets
  `connector_worker.worker = spec.get_metal_worker(kv_cache)`. No torch
  canonicalization. Scheduler-side behavior is fully inherited.

### 4.3 `MetalKVOffloadWorker(OffloadingWorker)`

One worker class serving both directions; `submit_store(job_id, src_spec,
dst_spec)` and `submit_load(job_id, src_spec, dst_spec)` make the direction
explicit — no `(src, dst)`-medium routing, no handler registration.

**Host pool**: per-layer numpy arrays, one per MLX cache array (K, V, and the
three TurboQuant side arrays when present), shaped
`(num_cpu_blocks, block_size_factor × block_size, heads, dim)` with a raw-bytes
(`uint16` view for bf16) dtype mapping. Plain numpy = pageable = the tier
semantics we want. No mmap, no pinning.

**Store (GPU→CPU)**, per layer: one gather
`mx.take(key_caches[L], mx.array(src_block_ids), axis=0)` → `np.asarray` →
strided write into the host pool rows. Reads go through the **current** list
entry `kv_cache.key_caches[L]`, whose lazy-graph provenance orders the read
after all pending in-place writes — this is the same reader-after-writer fence
the paged-attention kernel itself relies on (`sdpa.py:508-517`).

**Load (CPU→GPU)**, per layer: `mx.array(host_rows)` → index-assign
`cache[mx.array(dst_block_ids)] = ...` → rebind `kv_cache.key_caches[L]`,
mirroring the established write-then-rebind idiom so subsequent readers see
provenance.

**Completion model (M1)**: transfers execute synchronously inside
`submit_store`/`submit_load` (unified memory: a "transfer" is an on-package copy, and
stores are already deferred by the connector to post-sampling, when the graph
is materialized). Results queue for `get_finished()` with real
`transfer_size`/`transfer_time` so bandwidth stats work. `wait()` is a no-op
on an empty in-flight set. A follow-up can move stores to `mx.async_eval` or a
worker thread if profiling justifies it; the `OffloadingWorker` contract
(`submit_store`/`submit_load`/`get_finished`/`wait`) is already async-shaped.

**Block-size mapping**: the scheduler's GPU block ids index the MLX arrays'
leading dim directly (same `block_size`; the kernel-block translation in
`sdpa.py:122-160` is a read-time reshape and irrelevant here). One offloaded
(CPU) block = `block_size_factor` consecutive GPU blocks, exactly as in the
CUDA worker.

### 4.4 Worker hooks (`vllm_metal/v1/worker.py`)

- `initialize_from_config`: call
  `ensure_kv_transfer_initialized(vllm_config, kv_cache_config)` **before**
  `model_runner.initialize_kv_cache(...)` (mirrors
  `vllm/v1/worker/gpu_worker.py:591-606`). By this point the paged cache
  already exists (allocated in `determine_available_memory` →
  `setup_paged_attention`), so registration can happen immediately after.
- `get_kv_connector_handshake_metadata`: copied verbatim from
  `gpu_worker.py:554` — device-agnostic; returns `None` for this connector
  (base `get_handshake_metadata()` is not overridden), which
  `EngineCore.__init__` (`v1/engine/core.py:174-190`) handles fine. This fixes
  the M0 crash (`NotImplementedError` via `collective_rpc`).
- `shutdown`: `ensure_kv_transfer_shutdown()`.

### 4.5 Model-runner hooks (`vllm_metal/v1/model_runner.py`)

`MetalModelRunner` is standalone (no `GPUModelRunner`, no mixin) and splits a
step into `execute_model` (submit async MLX forward, return `None`) and
`sample_tokens` (sync, build `ModelRunnerOutput`). The connector lifecycle maps
onto that split; `OffloadingConnector.save_kv_layer`/`wait_for_layer_load`/
`wait_for_save` are documented no-ops, so no per-layer hooks are needed —
lucky, because the Metal forward is one monolithic MLX call.

- Top of `execute_model` (when `has_kv_transfer_group()`):
  `handle_preemptions`, `bind_connector_metadata(scheduler_output.kv_connector_metadata)`,
  `start_load_kv(None)` — loads run synchronously here, strictly before the
  forward is submitted, so the forward reads restored blocks.
  (`OffloadingConnector.start_load_kv` ignores its `forward_context` argument;
  we do not construct a vLLM `ForwardContext`.)
- No-work steps (`total_num_scheduled_tokens == 0`): run the same
  bind/start/get_finished sequence and return
  `ModelRunnerOutput.with_kv_conn_output_only(...)` — the connector must make
  progress on steps with no forward (mirrors `kv_connector_no_forward`,
  `vllm/v1/worker/kv_connector_model_runner_mixin.py:36`).
- End of step, in `sample_tokens` after sampling sync (and on the synchronous
  non-paged/pooling paths): `get_finished(finished_req_ids)` (which also queues
  deferred stores), `build_connector_worker_meta()`, assemble
  `KVConnectorOutput` onto the built `ModelRunnerOutput.kv_connector_output`,
  then `clear_connector_metadata()`. Stores thus execute on the *next* step's
  `start_kv_transfers`, after the previous graph materialized — cheap and
  correctly ordered.

All hooks are gated on `has_kv_transfer_group()`; without offloading flags the
code path is byte-for-byte unchanged.

### 4.6 Platform wiring (`vllm_metal/platform.py`, `vllm_metal/__init__.py`)

- Plugin registration: `KVConnectorFactory.register_connector("MetalOffloadingConnector", ...)`.
- `check_and_update_config`: when `kv_transfer_config` is set with
  `kv_connector == "OffloadingConnector"`, rewrite to the Metal subclass and
  inject `spec_name`/`spec_module_path` into `kv_connector_extra_config`.
  Reject configurations we can't serve yet (e.g. `lmcache` backend) with a
  clear error.

## 5. Correctness analysis

- **Read-after-write (stores)**: store gathers read through the latest rebound
  cache arrays; MLX's lazy graph serializes them after pending kernel writes.
  Additionally, stores are deferred by the connector to the step after
  sampling, when `mx.eval` has already materialized the writes.
- **Write-before-read (loads)**: loads run synchronously in `execute_model`
  before the forward graph is even built, and the rebind gives the forward's
  reads provenance through the scatter.
- **Logical-free vs physical reuse**: the scheduler only reuses a GPU block
  after its store job completed (`jobs_to_flush` fencing +
  `completed_jobs` in the worker meta) — same contract as CUDA; our worker
  reports completion only after the copy truly finished.
- **No worker failure path**: the connector worker asserts a successful
  submission, so errors raised in `submit_store`/`submit_load` propagate, and
  `OffloadingConnectorWorker.get_finished` asserts `success` (job failure is
  unsupported upstream); our worker must not report failures for recoverable
  conditions — validate sizes at registration time instead.
- **TurboQuant**: packed K/V + scale/zero arrays are all
  `(num_blocks, ...)`-leading, so the same gather/scatter per-array treatment
  works; the spec's byte sizing already accounts for the packed page size.
  Planned but validated after the fp16 path (M1 is fp16-first).
- **Excluded for now**: hybrid models with Mamba/GDN state (offloading spec
  only handles attention groups here), MLA cache, pipeline parallelism > 1.
  Guard with clear errors in `check_and_update_config`.

## 6. Milestones

- **M0 — boot**: worker RPC methods + `ensure_kv_transfer_initialized` +
  platform wiring + hooks with the real (synchronous) worker. Engine starts
  with `--kv-offloading-backend native --kv-offloading-size 2`; serves normally.
- **M1 — CPU tier verified**: with a tiny wired cache
  (`VLLM_METAL_MEMORY_FRACTION=0.05`) and repeated long prefixes,
  `vllm:prompt_tokens_by_source_total{source="external_kv_transfer"}` goes
  nonzero and greedy outputs are identical with/without offloading.
- **M2 — disk tier** (done, verified 2026-07-15): `MetalTieringOffloadingSpec`
  + `MetalSharedOffloadRegion` (anonymous in-process RAM behind a per-process
  registry — macOS has no tmpfs, and a file-backed mmap would write KV churn
  back to SSD through APFS; the platform hook enforces the single-process
  `uni` executor for tiering) with the worker's host pool carved as
  write-through strided numpy views over the region, so vLLM's `fs` tier
  reads/writes the same bytes via `create_kv_memoryview`. Verified: blocks
  cascade to disk (re-run on the in-process region: 10,295 `.bin` files /
  4.4 GB), a prefix evicted from both GPU and CPU tiers restores from disk
  (`external_kv_transfer` 0→2320, identical greedy output, 0.37s vs 1.55s
  cold), and — with `PYTHONHASHSEED=0` — the same prefix restores across a
  full server restart (fresh process, first request, 0.38s).

  Usage:

  ```bash
  PYTHONHASHSEED=0 VLLM_METAL_MEMORY_FRACTION=0.05 \
  VLLM_METAL_USE_PAGED_ATTENTION=1 \
  vllm serve <model> \
    --kv-offloading-backend native --kv-offloading-size <GiB> \
    --kv-transfer-config '{"kv_connector_extra_config":
      {"secondary_tiers": [{"type": "fs", "root_dir": "/path/to/kv-store"}]}}'
  ```

  `PYTHONHASHSEED` must be fixed for cross-restart / cross-instance reuse
  (block-content hash filenames are seeded by it; see
  `FileSystemTierManager`'s docstring).

  The `fs` tier resolves to `MetalFileSystemTierManager`
  (`vllm_metal/v1/kv_offload/fs_tier.py`), which layers macOS integrations
  and block-integrity checking over the upstream tier (lookup/store/load
  scheduling semantics are unchanged; the on-disk block format is not —
  see the CRC footer below):

  - Block files are `0o600` under a `0o700` root (KV blocks are
    conversation-derived data, and the deterministic hash filenames allow
    presence-testing of known prompts by anyone who can list the directory;
    upstream writes `0o644`).
  - Blocks land under `<root_dir>/blocks.noindex/` so Spotlight never
    indexes the churn, and `root_dir` gets a best-effort
    `tmutil addexclusion` so the store stays out of Time Machine backups and
    APFS local snapshots.
  - Block fds get `fcntl(F_NOCACHE)` — upstream's `O_DIRECT` degrades to
    buffered on macOS, which would flush useful page cache with multi-GB KV
    churn.
  - I/O threads set QoS: loads (request critical path)
    `QOS_CLASS_USER_INITIATED`, stores `QOS_CLASS_UTILITY` (E-core-friendly,
    doesn't compete with inference on P-cores).
  - **Block files carry an 8-byte little-endian CRC32 footer**
    (`<payload><crc>`), verified on every load. A torn write (power loss is
    likelier to tear with `F_NOCACHE`), bit rot, or a foreign/stale file
    fails the check *before* poisoned KV reaches the GPU cache; the bad
    file is deleted so the tier degrades to a clean miss, while transient
    errors (fd exhaustion, `EIO`) propagate without deleting. It is a
    corruption check, not an authentication code — store ownership
    (`0o600`/`0o700`) is the defense against hostile writers, and a
    per-deployment HMAC is the roadmap control for shared stores (§8).
  - **Lookups validate file size** (payload + footer), so truncated or
    foreign-layout files are misses up front, and a **failed-load negative
    cache** marks keys absent until re-stored — together these prevent a
    request-level livelock where the scheduler re-promotes a doomed block
    every step against upstream's per-request lookup cache.

  Note the store has no size cap or GC (upstream behavior): `root_dir` grows
  until manually cleared.
- **M3 — upstream PR** to vllm-project/vllm-metal (DCO, deterministic tests,
  docs page).

### Resume numerics — what "correct" means for restored prefixes

The restore path is **byte-exact**: per-row CRC instrumentation
(`VLLM_METAL_KV_OFFLOAD_DEBUG=1`) verifies stored bytes == loaded bytes ==
post-scatter GPU bytes. But an offload restore recomputes the prompt tail
(the tokens past the last full block) in a *different kernel batch shape*
than the original chunked prefill, so logits differ by fp16 epsilon
(~1e-3 nats measured). On prompts with near-tie next-token distributions
(a repetitive prompt showed a 0.045-nat top-2 margin) greedy argmax can
flip — deterministically — producing a different-but-legitimate
continuation. This reproduces **without the connector** (plain GPU
prefix-cache resume) and is inherent to restore-then-resume on any
backend, CUDA included. TurboQuant's discontinuous quantizers amplify the
wobble. Consequence for testing: compare restored output against a
**no-offload resume** (same batch shapes), never against the cold
prefill; byte-identical cold-vs-warm assertions are only valid for
high-margin (non-repetitive) prompts.

## 7. Test plan

- Unit: worker round-trip (store N blocks, zero the GPU blocks, load back,
  compare exact bytes) for fp16/bf16 and TurboQuant arrays; block_size_factor
  sub-block striding; spec sizing math vs `kv_cache_config`.
- CUDA parity: `tests/test_kv_offload_cuda_parity.py` pins the worker's
  host-pool byte placement to upstream's own `compute_sub_block_ptrs` (the
  CUDA DMA descriptor math, platform-independent) and asserts the
  scheduler-side surface is upstream code by function identity.
- fs tier: io round-trip under the CRC-footer format, corruption cases
  (truncated file, flipped payload bit → delete + clean miss; missing file →
  propagate without cleanup), size-aware store dedup, and an end-to-end
  manager test driving store → hit → corrupt → miss → failed-load negative
  cache (the livelock guard), mutation-checked against the lookup size check.
- Integration (slow/local): serve Qwen2.5-1.5B-4bit with FRACTION=0.05 +
  offloading, drive an evict-then-reuse workload, assert the
  `external_kv_transfer` counter and compare greedy completions against a
  no-offload run.

## 8. Future work

Ordered roughly by value. The first three need upstream conversations; the
rest are Metal-side.

- **Purgeable host pool** (upstream contract change). The maximal use of
  macOS for the "polite big tier": allocate the host pool as purgeable
  memory (`vm_allocate` + `VM_FLAGS_PURGABLE`, `vm_purgable_control`) so
  the kernel *discards* cold KV blocks under memory pressure instead of
  swapping them to SSD — swap that duplicates the disk tier's job with
  uncontrolled writes. Blocked today because a purge is a load that can
  fail, and `OffloadingConnectorWorker.get_finished` asserts success (job
  failure is unsupported upstream). With write-through tiering, a purged
  block could be re-promoted from disk if upstream grew a
  lookup-invalidate / load-retry path.
- **`madvise(MADV_FREE)` on evicted host blocks** (upstream hook needed).
  When the LRU evicts a CPU block its pages stay dirty-resident and macOS
  may swap them pointlessly. The scheduler-side manager doesn't notify the
  worker/region of eviction; an eviction callback (or consuming the
  existing KV events stream) would let the region release the row's pages.
  Rows are page-aligned in the shared region already.
- **fs-tier GC / disk budget** (upstream-shaped feature). The store grows
  without bound; a size cap with LRU file eviction, or a TTL sweep, is the
  missing retention policy. Also a data-retention question: persisted KV
  blocks are conversation-derived data. (Upstream `0o600` block files are
  worth proposing regardless.)
- **Transfer/compute overlap.** Transfers currently run synchronously on
  the engine thread (~18 ms per 66 MB store observed). `mx.async_eval` on
  the gather plus deferring the host copy to `get_finished` would
  approximate CUDA's stream overlap. Profile first: measured copy
  bandwidth (2.6–3.6 GB/s) is far below the machine's memory bandwidth,
  so there is also headroom in the copy path itself (fewer, larger
  gathers; avoiding per-array Python overhead).
- **Runtime memory-pressure response.** Startup is covered:
  `compute_safe_kv_budget` (`cache_policy.py`) clamps the paged-attention
  plan against live host memory — a hard cannot-fit cut plus a soft
  free-memory floor (`VLLM_METAL_MIN_FREE_FRACTION`, default 15% of RAM,
  min 4 GB; `VLLM_METAL_DISABLE_MEMORY_GUARD=1` opts out of the floor
  only). What remains is *runtime* response: subscribe to
  `kern.memorystatus_vm_pressure_level` (or the dispatch memory-pressure
  source) and shrink/purge the host pool under critical pressure instead
  of relying on the startup budget staying valid for the process
  lifetime.
- **Hybrid models and multi-rank.** Single KV cache group and single
  worker rank are explicitly guarded today; lifting them needs per-group
  worker plumbing (the CUDA worker's `group_sizes` path) and a real
  cross-process shared region (POSIX shm) respectively.
- **Encrypted-at-rest guidance and store authentication.** FileVault
  covers the default case; document pointing `root_dir` at an encrypted
  APFS volume for stronger isolation on shared machines. For stores that
  cannot rely on ownership alone (shared or network volumes), a
  per-deployment keyed HMAC in place of the CRC32 footer would
  authenticate blocks against hostile writers, not just detect
  corruption.
