# Prefill/Decode Disaggregation (Experimental)

!!! note
    This feature is **experimental** — a correctness-first bring-up of prefill/decode (PD) disaggregation on Apple Silicon, not a performance feature. It is validated end-to-end on Qwen3-0.6B, single Mac and across two Macs over Wi-Fi, with **token-identical** greedy output against non-disaggregated serving. Transfer is synchronous file I/O over a shared directory; see [Limitations](#limitations) and [Roadmap](#roadmap) before relying on it.

vllm-metal can run **disaggregated prefill**: one instance (the *producer*, `kv_role=kv_producer`) computes prompt KV and writes it out; a second instance (the *consumer*, `kv_role=kv_consumer`) serving the same model detects the stored prefix, loads the blocks into its own KV cache, and decodes from there without re-prefilling. Both roles are played by a single connector class, `MetalFileConnector`, plugged in through vLLM's standard v1 KV-transfer config. To our knowledge this is the first PD-disaggregated serving path on a pure Apple-Silicon cluster (prior public Mac PD work mixes a DGX Spark with Macs, not Macs only).

## Architecture

The transfer plane is a **shared directory** (local FS, NFS, or sshfs). Each stored prefix is one folder, keyed by the sha256 of the block-aligned prompt prefix:

```
<shared_storage_path>/<sha256(prompt tokens + mm hashes)>/
    layer_0000.safetensors   # {"key": [n_blocks, block_size, kv_heads, head_dim],
    layer_0001.safetensors   #  "value": [...]}  — mx.array block-pool slices
    ...
    done                     # marker written only after every layer file is complete
```

The `done` marker guarantees a consumer never reads a half-written prefix; a lookup that finds no complete entry simply misses and the request prefills locally (standard vLLM fallback).

- **`vllm_metal/kv_connector.py`** — `MetalFileConnector` implements the full `KVConnectorBase_V1` interface of vLLM 0.28.0 (`get_num_new_matched_tokens`, `update_state_after_alloc`, `build_connector_meta`, `start_load_kv`, `save_finished_requests`, …). One class plays both roles, selected by `kv_role`.
- **`vllm_metal/v1/model_runner.py`** — the worker-role connector is initialized lazily after `initialize_kv_cache`, loaded externally via `KVConnectorFactory` + `kv_connector_module_path` (zero changes to the vllm core repo). Blocks are bulk-loaded before the forward; after the async forward materializes (in the `sample_tokens` path), finished requests are bulk-stored. In PD mode the decode pipeline gate is forced ineligible, keeping per-step synchronous sampling.
- **`vllm_metal/v1/worker.py`** — `get_kv_connector_handshake_metadata` returns `None`: a file-based connector needs no out-of-band handshake, and returning `None` keeps EngineCore's startup handshake-collection path working.
- **Scope guard** — single-KV-group models only (dense MHA/GQA). Hybrid linear-attention models raise `NotImplementedError` explicitly. The connector does not declare `SupportsHMA`, so PD requires `--disable-hybrid-kv-cache-manager`.

## Quick start (single Mac, two processes)

```bash
STORE=$HOME/tmp/pd-kvshare
REV=c1899de289a04d12100db370d81485cdf75e47ca   # validated revision

# 1. Prefill instance (producer): computes prompt KV, stores it to $STORE.
GLOO_SOCKET_IFNAME=lo0 VLLM_METAL_USE_PAGED_ATTENTION=1 \
vllm serve Qwen/Qwen3-0.6B --revision "$REV" \
  --max-model-len 4096 --gpu-memory-utilization 0.35 \
  --disable-hybrid-kv-cache-manager --port 8100 \
  --kv-transfer-config "{\"kv_connector\":\"MetalFileConnector\",\"kv_connector_module_path\":\"vllm_metal.kv_connector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"shared_storage_path\":\"$STORE\"}}"

# 2. Decode instance (consumer, second terminal): same model/revision,
#    detects stored prefixes and loads them instead of prefilling.
GLOO_SOCKET_IFNAME=lo0 VLLM_METAL_USE_PAGED_ATTENTION=1 \
vllm serve Qwen/Qwen3-0.6B --revision "$REV" \
  --max-model-len 4096 --gpu-memory-utilization 0.35 \
  --disable-hybrid-kv-cache-manager --port 8200 \
  --kv-transfer-config "{\"kv_connector\":\"MetalFileConnector\",\"kv_connector_module_path\":\"vllm_metal.kv_connector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"shared_storage_path\":\"$STORE\"}}"
```

Both instances must serve the same model and revision (identical KV layout). On a healthy PD request the producer logs the stored blocks and the consumer reports the prompt as an external hit and loads the blocks (`External` hit + `Loaded`).

A minimal stdlib-only demo proxy ties the two instances into one endpoint: it forwards every completion **twice** — first to the producer with `max_tokens=1` (which computes and stores the prompt KV), then to the consumer with the original body (which loads the stored KV and generates the answer). The client only ever sees the consumer's response.

## Quick start (two Macs)

Run the same two commands, one per Mac. The two sides must see the same storage, either way:

- **Shared filesystem** — mount one directory on both Macs (NFS or sshfs) and point `shared_storage_path` at it on both sides. Completeness is guarded by the `done` marker, so the consumer never reads a partial entry.
- **rsync watcher (what we validated)** — without a shared FS, a two-phase rsync-over-ssh loop replicates the store directory from the prefill Mac to the decode Mac: first pass syncs everything except `done` markers, second pass syncs only the markers, so a marker never lands before its payload. Observed throughput: ~72–176 MB/s over Wi-Fi LAN.

Our cross-Mac validation ran prefill on an M4 Max 128 GB and decode on an M3 Max 96 GB over Wi-Fi.

## Verification

Validated configuration: **Qwen3-0.6B @ `c1899de`, `block_size=16`, 28 KV layers**.

- **Unit tests** (`tests/test_kv_connector.py`, 4/4 green; ruff + mypy clean): store→load block-level roundtrip into different block ids; load without a `done` marker raises `FileNotFoundError`; scheduler miss→hit after a store; `update_state_after_alloc` + `build_connector_meta` state cleanup.
- **Single Mac, two processes** (190-token prompt): the producer stored 11 blocks × 28 layers (19 MB); the consumer took the external hit, loaded the blocks, and its greedy output matched the non-disaggregated ground truth **28/28 tokens, identical**.
- **Two Macs** (M4 Max 128 GB prefill → Wi-Fi LAN → M3 Max 96 GB decode): same 11-block hit and load; output again token-identical to ground truth.

Token identity is the core of the check: under greedy decoding, any corruption, dtype mismatch, or block-layout error in the transferred KV would diverge the output stream immediately. Matching every token is the strongest practical evidence that the handoff is lossless. To reproduce: (1) serve one instance without `--kv-transfer-config`, send a greedy request, save the output; (2) bring up the producer/consumer pair and send the same prompt through the proxy; (3) diff the token streams — they must be identical.

### Benchmark (3,400-token prompt, 48 generated tokens)

| Path | Time |
|---|---|
| Monolithic on consumer (M3 Max: full prefill + decode) | 1.60 s |
| PD: prefill on producer (M4 Max) | 1.17 s |
| PD: KV ship (371 MB = 212 blocks × 28 layers, rsync) | 5.12 s |
| PD: decode on consumer (load 212 blocks + 48 tokens) | **0.59 s** |

The decode instance's work drops **2.7×** (1.60 s → 0.59 s) — that is the disaggregation effect: the consumer no longer prefills. End-to-end time is currently dominated by the demo rsync transport, not by the connector; the numbers characterize the *file* transfer plane, not the ceiling of PD on Apple Silicon.

## Design notes (anticipated review questions)

- **Why store only the block-aligned prefix (the `len - 1` alignment)?** The consumer must compute at least the final prompt token itself, so the transferable prefix excludes the last (possibly partial) block. This mirrors `ExampleConnector`'s key semantics; the consumer's forward covers the unaligned tail, and greedy output was verified token-identical against monolithic serving.
- **Why bulk `save_finished_requests` instead of per-layer `save_kv_layer`?** The Metal path has no per-attention-op hookpoint (MLX attention wrappers do not expose the layer boundary the CUDA custom-op path does), so the plugin's model runner performs one bulk store after the async forward materializes. This is a deliberate runner-side integration, trading the async overlap of per-layer saves for a minimal, auditable diff; the roadmap's layered streaming restores the overlap.
- **Integrity.** Every store carries a `manifest.json` (layer count, block geometry, dtype); loads validate it and fail fast on mismatch — a wrong-shaped or stale store can never be scattered into the paged cache. `max_stored_prefixes` (extra config) optionally prunes oldest stores.

## Limitations

- **Synchronous file transfer.** Store/load sit on the forward path. Performance is explicitly not the goal of this bring-up; correctness is.
- **Demo-grade cross-machine transport.** The validated cross-Mac setup used a two-phase rsync-over-ssh watcher (~72 MB/s observed over Wi-Fi). NFS/sshfs shares work; nothing here is fast yet.
- **Short prompts don't transfer.** Prefixes shorter than `block_size + 1` tokens are not stored/loaded.
- **All-or-nothing hits.** The transfer key is the whole block-aligned prompt, so there is no prefix sharing across different-length prompts: a shorter prompt does not hit a longer one's store. Fine for v1; prefix-granular keys are future work. `cache_salt` is folded into the key; LoRA adapters are not yet (same prompt under different adapters must not share stores today — use separate storage paths per adapter).
- **Chunked prefill.** Only requests scheduled as a single full chunk are stored (PoC limitation).
- **Single KV cache group only.** Dense MHA/GQA models. Hybrid linear-attention models raise `NotImplementedError`; MLA is not supported.
- **HMA disabled.** The connector does not declare `SupportsHMA`; PD runs require `--disable-hybrid-kv-cache-manager`.

## Roadmap

- **Async, layered streaming** — store/load each layer as the forward progresses, instead of bulk transfer after materialization.
- **TCP connector** — replace the file plane with direct socket transfer (NIXL TCP or similar). The connector interface stays; only the transport swaps.
- **Hybrid models** — lift the single-KV-group restriction for hybrid linear-attention architectures.
- **Portability** — the on-disk format is plain per-layer safetensors block-pool slices; the same connector pattern should port to other MLX serving stacks.
