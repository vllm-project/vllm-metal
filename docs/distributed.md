# Distributed Inference with Ray

!!! note
    For normal single-Mac serving you don't need any of this — use the default in-process executor. The Ray executor and pipeline parallelism below are for **multi-Mac** serving (running one model across several Macs). Single-node serving, the pipeline-parallel forward, and a two-Mac end-to-end run (Qwen3-0.6B over Thunderbolt) are validated; multi-Mac serving is still new — **verify it for your own models and setup before relying on it.**

vllm-metal can run under vLLM's **Ray distributed executor**, placing each Apple-Silicon worker as a Ray actor. This is the groundwork for multi-Mac serving; today the **single-node** path (one Mac, `--tensor-parallel-size 1`) is supported and validated. **Pipeline parallelism** — splitting a model across stages — is numerically validated on a single node (see [Pipeline parallelism](#pipeline-parallelism)) and has served a model end-to-end across two Macs over Thunderbolt (see [Limitations](#limitations)).

Apple GPUs are not a Ray-recognized accelerator type (unlike CUDA or TPU), so each node advertises a **custom Ray resource named `mlx`**, and vLLM's executor places one worker per `mlx` unit.

## Requirements

- `ray` installed alongside vllm-metal: `uv pip install ray` (or `pip install ray`).
- Each node started with one `mlx` resource (one Apple GPU per Mac).
- The Ray node IP must match the address vLLM resolves via `get_ip()` — do not mix loopback and LAN addresses.

## Quick start (single node)

```bash
# 1. Start a Ray head node advertising the Apple GPU as the "mlx" resource.
#    Pin the node IP to the address vLLM uses so placement-group binding matches.
IP=$(python -c "from vllm.utils.network_utils import get_ip; print(get_ip())")
ray start --head --node-ip-address="$IP" --resources='{"mlx": 1}'

# 2. Serve through the Ray executor (connects to the running cluster).
#    Plain `--distributed-executor-backend ray` uses the default Ray V2 executor.
RAY_ADDRESS=auto vllm serve Qwen/Qwen3-0.6B \
  --distributed-executor-backend ray \
  --tensor-parallel-size 1

# 3. Verify generation runs through Ray.
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
# → " Paris. The capital of Italy is Rome. The capital of Spain is Madrid."

# Tear down when done.
ray stop
```

If a node isn't advertising the `mlx` resource, the engine can't place workers: the Ray executor logs `No available node types can fulfill resource request {'mlx': 1.0, ...}` and hangs while creating the placement group. (A separate `current platform cpu does not support ray` error instead means the Metal plugin isn't active or `ray_device_key` is unset — not a missing resource.)

On a healthy boot, each worker logs `vllm_metal: patched Ray V2 worker get_node_and_gpu_ids on RayWorkerProc (Apple-GPU custom Ray resource)` — confirming the custom-resource override fired inside the Ray actor.

## Quick start (two Macs over Thunderbolt)

Run one model split across two Macs with [pipeline parallelism](#pipeline-parallelism): **stage 0** (first layers) on Mac A, **stage 1** (last layers + sampling) on Mac B. Ray is the control plane; the cross-stage activations travel over the **MLX ring** on a direct **Thunderbolt cable**. That high-bandwidth, low-latency link is what makes PP across machines worthwhile — Wi-Fi / Ethernet is too slow to serve over, so Thunderbolt is the supported transport.

!!! note
    Multi-Mac serving is new — validated end-to-end on Qwen3-0.6B; start with that small model to check the plumbing, then scale up. Both Macs must have the model cached, the Thunderbolt bridge reachable, and each Mac's firewall must allow the MLX ring ports (`32323`/`32324` by default; stage *r* uses `base + r`). If those ports are busy — an `mlx.launch` job, a quick restart still in `TIME_WAIT`, or a second PP job — set `VLLM_METAL_RING_BASE_PORT` to the same base on **every** node to shift the whole block.

!!! note
    A multi-node Ray cluster on **macOS** needs two extra env vars on every node, both exported in the commands below: `RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1` (macOS clustering is gated behind it), and — only if the two Macs aren't on the identical Python build — `RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=minor` (Ray otherwise refuses to join nodes whose Python differs even at the *patch* level; the same minor version is wire-compatible).

Connect the two Macs with a Thunderbolt / USB4 cable — macOS auto-creates a "Thunderbolt Bridge" interface — and give it a static IP on each (same subnet, different last octet):

```bash
# Mac A
sudo networksetup -setmanual "Thunderbolt Bridge" 10.0.0.1 255.255.255.0
# Mac B
sudo networksetup -setmanual "Thunderbolt Bridge" 10.0.0.2 255.255.255.0
```

From Mac A, confirm the cable is up: `ping -c3 10.0.0.2`.

gloo (vLLM's control plane) advertises whatever each node's **hostname** resolves to — and a macOS `.local` hostname resolves to loopback, which can't be reached from the other Mac. Map each node's hostname to its bridge IP, and leave `GLOO_SOCKET_IFNAME` unset:

```bash
# Mac A
sudo scutil --set HostName maca && echo "10.0.0.1  maca" | sudo tee -a /etc/hosts
# Mac B
sudo scutil --set HostName macb && echo "10.0.0.2  macb" | sudo tee -a /etc/hosts
# verify on each — must print the bridge IP, not 127.0.0.1:
python -c "import socket; print(socket.gethostbyname(socket.gethostname()))"
```

**Mac A** — start the Ray head, pinned to its Thunderbolt IP:

```bash
source .venv-vllm-metal/bin/activate
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1            # multi-node Ray on macOS
export RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=minor   # only if the Macs' Python patch versions differ
VLLM_HOST_IP=10.0.0.1 ray start --head \
  --node-ip-address=10.0.0.1 --resources='{"mlx": 1}'
```

**Mac B** — join over Mac A's Thunderbolt IP:

```bash
source .venv-vllm-metal/bin/activate
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
export RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=minor   # only if the Macs' Python patch versions differ
VLLM_HOST_IP=10.0.0.2 ray start --address=10.0.0.1:6379 \
  --node-ip-address=10.0.0.2 --resources='{"mlx": 1}'
```

**Mac A** — serve across both stages:

```bash
RAY_ADDRESS=auto VLLM_HOST_IP=10.0.0.1 VLLM_METAL_USE_PAGED_ATTENTION=1 \
  vllm serve Qwen/Qwen3-0.6B \
    --distributed-executor-backend ray \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 1 \
    --no-async-scheduling
```

`--no-async-scheduling` is required for PP — the first stage has no sampler and rebuilds the token stream from the scheduler, which async scheduling would leave empty (the engine fails loud if you omit it).

On a healthy boot each worker logs its stage — `Pipeline stage 0/2 (is_first=True, is_last=False)` and `Pipeline stage 1/2 (is_first=False, is_last=True)` — and the MLX ring bootstrap lists both Macs' Thunderbolt IPs.

**Query** (from Mac A):

```bash
curl -s http://10.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
```

Tear down with `ray stop` on **both** Macs, then revert the bridge: `sudo networksetup -setdhcp "Thunderbolt Bridge"`. Once the small model works end to end, try larger models across the two Macs (each stage still loads the full model first — see [Limitations](#limitations)).

`VLLM_HOST_IP` is the load-bearing piece: it makes vLLM's `get_ip()` return the Thunderbolt address, so the cross-stage hand-off forms over the cable.

## How it works

- `MetalPlatform` sets `ray_device_key = "mlx"` and `device_control_env_var = "VLLM_METAL_VISIBLE_DEVICES"`, so vLLM's Ray executor takes its generic custom-resource placement path (the same one TPU uses) instead of the CUDA `num_gpus` path.
- A compatibility shim overrides the Ray worker's `get_node_and_gpu_ids` to read the assigned `mlx` resource — Ray never lists custom resources in `get_accelerator_ids()` — installed in each worker via a Ray `worker_process_setup_hook`.

## Pipeline parallelism

Pipeline parallelism (PP) splits the model's transformer layers into contiguous
stages — one worker process per stage — and pipes the hidden state from one stage
to the next. It is the natural fit for multi-Mac serving: unlike tensor
parallelism (an all-reduce every layer), PP sends a single activation per stage
boundary, so it tolerates a Thunderbolt / Ethernet link between machines.

**Design.** The mlx_lm model files are untouched. vLLM's executor (Ray, or the
`mp` executor for a single node) owns the *control plane* — spawning one ranked
worker per stage. The *data plane* — the cross-stage activation hand-off — runs
over MLX's own `mx.distributed` **ring** backend (point-to-point `send` / `recv`),
not Ray. Rank 0 is the first stage (it embeds the tokens); rank `N-1` is the last
(final norm, head, sampling). Each stage owns a contiguous layer slice and only
the last stage produces logits. Tensor parallelism must be 1, so the global rank
equals the pipeline-stage index.

**Numerical validation (single node).** `tools/pp_parity_check.py` runs the PP
forward across `N` ring processes and compares the logits against a single-process
reference. Because every stage runs identical ops on its slice and the hidden
state crosses the wire unchanged, the result is **bit-exact**:

```bash
source .venv-vllm-metal/bin/activate

# 1. Reference logits (single process; writes /tmp/pp_ref_<model>.npy).
python tools/pp_parity_check.py Qwen/Qwen3-0.6B

# 2. Two-stage pipeline over the MLX ring, compared against the reference.
mlx.launch -n 2 --backend ring tools/pp_parity_check.py Qwen/Qwen3-0.6B
# → PARITY PASS max_abs_diff=0.000e+00   (validated on Qwen3-0.6B / 1.7B / 4B)
```

## Limitations

- **Peak memory = full model per node.** Each stage loads the whole model before dropping its non-owned layers, so Phase 0 splits compute, not load-time memory — it can't serve a model too large for one Mac.
- **Co-located stages oversubscribe the KV budget.** Each stage applies `VLLM_METAL_MEMORY_FRACTION` to the whole device independently — neither knows the other exists — so two stages on one Mac claim roughly twice the fraction. Lower it when stacking. Separate Macs are unaffected.
- **Synchronous scheduling required.** Run with `--no-async-scheduling` (the engine fails loud otherwise).
- **TP=1 only.** PP+TP is rejected; tensor parallelism (`--tensor-parallel-size > 1`) is not implemented.
- **Model support.** YOCO / hybrid / MLA / pooling / VLM / non-paged / speculative decoding / LoRA are rejected; other shapes (sliding-window, MoE) are untested.

## Data parallelism

Data parallelism (DP) runs **N independent full-model replicas**, one per Mac, behind a single endpoint, and load-balances requests across them. Unlike pipeline parallelism it is a pure **throughput** scale-out: each replica holds the whole model, so DP does **not** serve a model larger than one Mac — use it only for a model that already fits one Mac. On a fixed cluster the two are mutually exclusive uses of the same nodes: **DP = more requests/sec for a model that fits; PP = one model split across nodes for longer context / a compute split.**

Dense DP needs no cross-device collective — vLLM runs each replica as a fully independent engine placed on the `mlx` resource, and the request load balancer is upstream and platform-agnostic. Only the validated **dense + Ray DP backend + one replica per node + internal load balancer** shape is supported; everything else fails fast at config time (see the limitations below).

**Serving (two Macs).** Bring up the Ray cluster exactly as for [pipeline parallelism](#pipeline-parallelism) — `ray start` on both Macs with `--resources='{"mlx":1}'`, the macOS cluster env vars, and a per-node `VLLM_HOST_IP` — then serve with DP instead of PP:

```bash
# Mac A (head): one full replica per Mac, Ray DP backend, internal LB.
RAY_ADDRESS=auto VLLM_HOST_IP=10.0.0.1 VLLM_METAL_MEMORY_FRACTION=0.5 \
  vllm serve mlx-community/Qwen3-8B-4bit \
    --max-model-len 8192 \
    --data-parallel-size 2 \
    --data-parallel-backend ray \
    --data-parallel-size-local 1 \
    --data-parallel-address 10.0.0.1
```

- `--data-parallel-backend ray` is **required**: the default `mp` backend only spawns local subprocesses and cannot place a replica on a second Mac (it would silently overcommit one Mac).
- `--data-parallel-size-local 1`: one Apple GPU per Mac means one replica per node.
- `--data-parallel-address <head-ip>`: pin the DP master to the Ray head's IP so placement finds it. The Ray DP backend otherwise follows `get_ip()` / `VLLM_HOST_IP`, so set it explicitly to avoid a mismatch.

On a healthy boot each Mac logs `patched Ray V2 worker get_node_and_gpu_ids ...` and an `EngineCore` actor is placed on each node IP.

**Design.** vLLM owns the whole DP control plane (replica placement, the DP coordinator, the request load balancer); `MetalPlatform` only relaxes admission to the supported shape. One Metal-specific detail: Ray honours a `worker_process_setup_hook` only from the **job** runtime_env (`ray.init`), and the DP engine manager connects to Ray without forwarding it — so vllm-metal registers the Apple-GPU worker patch at the job level itself before the engine connects (`MetalPlatform._register_dp_ray_worker_setup_hook`); otherwise the per-replica `RayWorkerProc` would `KeyError` on the custom `mlx` resource.

DP helps **under concurrency**, not single-stream latency, and stays below the 2× ideal because the head Mac also runs the API server, the DP coordinator, and the load balancer alongside its own replica; adding more replica nodes amortizes that head overhead. Measure your own throughput with `vllm bench serve`.

**Limitations.**

- **Capacity is unchanged.** Each replica loads the full model; DP does not serve a model larger than one Mac (that needs a sharded load, which is not yet available).
- **Dense models only.** MoE DP (expert-parallel all-to-all) is rejected — MLX has no `all_to_all` collective.
- **Requires a running Ray cluster.** DP registers the worker hook by initializing Ray itself; it fails loud if no cluster is reachable, or if Ray was already initialized by something else (do not pre-`ray.init` before serving).
- **Validated at 2 Macs.** More nodes (one replica each) should work but are untested.
- **Rejected combinations** (fail fast at config time): DP+PP, DP+TP, DP+MoE, DP+multimodal (the multimodal tensor-IPC path is DP=1 only), DP+speculative-decoding, DP+LoRA, DP+STT, `--data-parallel-external-lb` / `--data-parallel-hybrid-lb`, and any `--data-parallel-size-local` other than 1 (including the external-DP sentinel 0).
