# KV Cache Offloading

Offloading keeps KV blocks evicted from the Metal KV cache, in a host memory
pool and optionally on disk. A returning prefix, or a restarted server, then
restores those blocks instead of recomputing them. Offloading is off by
default and uses vLLM's own flags.

## Quick start

```bash
# Host pool only
vllm serve Qwen/Qwen3-8B --kv-offloading-size 8

# Host pool plus a 50 GiB disk tier
vllm serve Qwen/Qwen3-8B \
  --kv-offloading-size 8 \
  --kv-transfer-config '{"kv_connector_extra_config": {"secondary_tiers":
    [{"type": "fs", "root_dir": "/path/to/kv-store", "max_size_gib": 50}]}}'
```

## Options

| Flag | Description |
|---|---|
| `--kv-offloading-size N` | Host pool size in GiB. Enables offloading. Optional when `--kv-transfer-config` names the connector: the pool then defaults to two `--max-model-len` requests of KV, and the startup log reports the size. |
| `--kv-offloading-backend` | `native` (default). `lmcache` is refused. |
| `--kv-transfer-config` | Secondary tiers, as JSON. Only `fs` is supported. |

Keys for an `fs` tier:

| Key | Default | Description |
|---|---|---|
| `type` | required | `fs` |
| `root_dir` | required | Where block files are stored. |
| `max_size_gib` | 10% of the volume | Disk cap in GiB. `0` means no cap. If unset and the disk has less free space than 10% of the volume, startup fails. |

Without `--kv-offloading-size`, the pool defaults to the KV of two
`--max-model-len` requests, rounded up to whole blocks. A full pool does not lose
a block, the scheduler retries the store on the next step; two requests is the
smallest pool with no retries at concurrency 4 in the #1037 measurements. The
default is capped at a quarter of the KV budget (after weights and
activations), or one host chunk if that is larger, and leaves room for one
`--max-model-len` request. With `--max-model-len -1` or
`--num-gpu-blocks-override` no request is reserved. The startup log says when
the pool is changed. Pass the flag to size it yourself.

The host pool comes out of `--gpu-memory-utilization` (see
[Configuration](configuration.md#kv-cache-memory-settings)), which is the total
Metal budget including weights. A larger pool means a smaller KV cache, not
extra memory. The startup memory breakdown shows it as `kv_offload_pool=`. A
pool set with `--kv-offloading-size` that is larger than the budget fails at
startup, after the weights load, with the breakdown and the fix. 8 GiB was
enough for the Qwen3-32B benchmark in #737.

The pool takes memory only as chunks are stored. Level 1 sleep frees it (see
[Sleep mode](configuration.md#sleep-mode)).

## Limits

Only uniform full-attention models are supported (GQA, MHA, MQA), for example
Qwen3 dense models. Refused at startup:

- Sliding-window models, MLA models, and models whose KV cache splits into
  more than one group (hybrid models).
- Draft-model speculative decoding.
- Pooling and speech-to-text models.
- Pipeline and data parallelism, and any executor other than `uni`.
- `--kv-offloading-size` combined with a different KV connector.
- `canonical_layout` in `kv_connector_extra_config`. Metal does not write the
  canonical layout.

## Reuse across restarts

Block files are named by content hash. With the default
`--prefix-caching-hash-algo sha256` (or `sha256_cbor`) the hash seed is fixed,
so a restarted server finds the blocks written before. Stores are not shared
with vLLM on other platforms.

With `xxhash` or `xxhash_cbor` the seed is random per process, so a restarted
server cannot find earlier blocks unless `PYTHONHASHSEED` is set; the server
warns at startup. Every server sharing a store must use the same value.

## KV events for routing

A KV-aware router can track which blocks each server holds, including blocks
on disk. Turn on KV cache events with `--kv-events-config`
(`"enable_kv_cache_events": true` plus a publisher). The disk tier then also
publishes its stores and evictions. `enable_kv_events` on the tier is set for
you. Setting it on the tier without the global switch logs a warning, because
the tier would publish nothing.

## Operational notes

- **Disk cap.** When a store would cross the cap, the least recently stored or
  restored blocks are deleted first. Files with a load or store in flight are
  never deleted. The cap is per process, so give each server its own
  `root_dir`. A Time Machine local snapshot can hold deleted block files until
  macOS reclaims the space, which it does when the disk runs low.
- **Changed weights.** The store is keyed by model name and dtype, not by
  weights. Delete `root_dir` after replacing a checkpoint under the same name.
- **Permissions.** The store directory is created `0700`, which keeps other
  users out of the block files. Prompt content can be recovered from them.
- **Backups.** Blocks go under a `.noindex` directory so Spotlight skips them.
  Time Machine is not excluded for you, because `tmutil addexclusion` can block
  startup for several seconds. Run it yourself if needed:
  `tmutil addexclusion /path/to/kv-store`.
