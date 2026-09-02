# KV Cache Offloading

vllm-metal supports vLLM's KV cache offloading. Blocks evicted from the wired
Metal KV cache spill to host memory and, optionally, to disk, so a prefix that
no longer fits can be restored instead of recomputed. The flags are vLLM's own,
so a command that works on CUDA works here.

Two things prefix caching cannot do and offloading can: serve a prefix after it
has been evicted, and serve one after the server restarts.

## Quick Start

```bash
VLLM_METAL_USE_PAGED_ATTENTION=1 vllm serve mlx-community/Qwen2.5-32B-Instruct-4bit \
  --kv-offloading-size 32
```

That gives the host-memory tier. On Apple Silicon it is the same physical RAM
as the wired cache, so it buys capacity beyond the wired cap rather than a
faster medium.

For persistence, which is where the benefit is largest, add a disk tier:

```bash
PYTHONHASHSEED=0 \
VLLM_METAL_USE_PAGED_ATTENTION=1 vllm serve mlx-community/Qwen2.5-32B-Instruct-4bit \
  --kv-offloading-size 32 \
  --kv-transfer-config '{"kv_connector_extra_config":
    {"secondary_tiers": [{"type": "fs", "root_dir": "/path/to/kv-store"}]}}'
```

Paged attention is required. Offloading is off by default.

## Configuration

| Flag | Description |
|---|---|
| `--kv-offloading-size N` | Host pool size in GiB. Enables offloading. |
| `--kv-offloading-backend` | Must be `native`. |
| `--kv-transfer-config` | Secondary tiers, as JSON. Only `fs` is supported. |

Tier keys:

| Key | Default | Description |
|---|---|---|
| `type` | required | `fs`. `p2p` needs NVLink, `obj` needs NIXL, which has no macOS build. |
| `root_dir` | required | Where block files live. Nothing removes them; see Disk usage. |
| `enable_kv_events` | auto | Set for you when KV events are on globally. |

### `PYTHONHASHSEED` is required for reuse across restarts

Block filenames are content hashes, and Python seeds its hash per process.
Without `PYTHONHASHSEED=0` a restarted server cannot find what the previous one
wrote, so the disk tier silently gives you nothing on the case it exists for.
The server warns at startup when a persistent tier is configured and the seed
is unset.

## Sizing

**The host pool should be larger than the wired KV cache.** If it is smaller,
the pool's own LRU drops blocks before the wired cache would have re-requested
them, so restores mostly miss and the tier costs more than it saves. The
server logs a warning when this happens and no disk tier is configured.
Behind a disk tier a small pool is the expected shape.

The wired cache size is `VLLM_METAL_MEMORY_FRACTION` of the recommended working
set, minus the model weights. It is printed at startup as
`max_tokens_cached=`. Note the tension: at a high memory fraction there is no
room left for a pool larger than the cache, and the disk tier is then the only
secondary tier that adds real capacity.

The host pool is pageable memory and comes out of the same RAM as the weights
and the wired cache. A pool above half of physical RAM is refused at startup,
because it would swap rather than fail cleanly.

## Disk usage

**Nothing evicts block files.** The store grows to whatever the workload
touches and stays there until you delete `root_dir` yourself. A workload with
no prefix reuse writes every evicted block and reads none of them back; on a
32B model that reached 84 GB in one benchmark run.

Blocks live under a `blocks.noindex` subdirectory so Spotlight does not index
them, and files are `0600` under a `0700` root because prompt content is
recoverable from them. The store is not excluded from Time Machine for you.
Run `tmutil addexclusion /path/to/kv-store` if it is on a backed-up volume.

## KV-aware routing

A KV-aware router places requests by knowing which instance already holds
a prefix. It learns that from the `BlockStored` events the offload tier
publishes, so the events have to be turned on:

```bash
PYTHONHASHSEED=0 \
VLLM_METAL_USE_PAGED_ATTENTION=1 vllm serve <model> \
  --kv-offloading-size 32 \
  --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq",
    "endpoint": "tcp://*:5557"}' \
  --kv-transfer-config '{"kv_connector_extra_config":
    {"secondary_tiers": [{"type": "fs", "root_dir": "/path/to/kv-store"}]}}'
```

`enable_kv_events` on the tier is set automatically when
`--kv-events-config` enables events globally. Setting it on the tier *without*
enabling events globally is refused at startup, because the tier would publish
nothing and the router would silently route as if this instance held no
prefixes.

## Not supported

Rejected at startup with a specific message, rather than failing later.

- Hybrid and sliding-window models. A single uniform full-attention KV cache
  group only.
- Draft-model speculative decoding, which puts the draft KV in the same cache
  group.
- Multiple workers. Offloading needs the single-process executor, because the
  host pool is shared between the scheduler and the worker in one process.
- The `p2p` and `obj` tiers.

## Known limitations

Block files carry no checksum, matching upstream's on-disk format so a store
written here is readable by an upstream `fs` tier and the reverse. A torn write
or bit rot therefore produces a right-sized file whose contents are wrong, and
it is restored without complaint. Only truncation is caught.

Deleting block files does not immediately return disk space, because APFS local
snapshots are volume-wide and have no per-path exclusion.
