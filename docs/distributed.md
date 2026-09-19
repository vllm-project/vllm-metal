# Distributed Inference with Ray

!!! note
    For normal single-Mac serving you don't need any of this — use the default in-process executor. The Ray executor and distributed modes below are for **multi-Mac** serving (running one model across several Macs). Single-node serving, the pipeline-parallel forward, and a two-Mac end-to-end run (Qwen3-0.6B over Thunderbolt) are validated; multi-Mac serving is still new — **verify it for your own models and setup before relying on it.**

vllm-metal can run under vLLM's **Ray distributed executor**, placing each Apple-Silicon worker as a Ray actor. The single-node path remains the default. Multi-Mac modes include [pipeline parallelism](#pipeline-parallelism), with numerical and hardware validation, and the scoped [GPT-OSS TP2 path](#gpt-oss-tensor-parallelism).

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

On a healthy boot, each worker logs `vllm_metal: patched Ray V2 worker get_node_and_physical_gpu_ids on RayWorkerProc (Apple-GPU custom Ray resource)` — confirming the custom-resource override fired inside the Ray actor.

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
RAY_ADDRESS=auto VLLM_HOST_IP=10.0.0.1 \
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

Tear down with `ray stop` on **both** Macs, then revert the bridge: `sudo networksetup -setdhcp "Thunderbolt Bridge"`. Once the small model works end to end, try larger models across the two Macs.

`VLLM_HOST_IP` is the load-bearing piece: it makes vLLM's `get_ip()` return the Thunderbolt address, so the cross-stage hand-off forms over the cable.

## How it works

- `MetalPlatform` sets `ray_device_key = "mlx"` and `device_control_env_var = "VLLM_METAL_VISIBLE_DEVICES"`, so vLLM's Ray executor takes its generic custom-resource placement path (the same one TPU uses) instead of the CUDA `num_gpus` path.
- A compatibility shim overrides the Ray worker's `get_node_and_physical_gpu_ids` to read the assigned `mlx` resource — Ray never lists custom resources in `get_accelerator_ids()` — installed in each worker via a Ray `worker_process_setup_hook`.

## Pipeline parallelism

Pipeline parallelism (PP) splits the model's transformer layers into contiguous
stages — one worker process per stage — and pipes the hidden state from one stage
to the next. It is the natural fit for multi-Mac serving: unlike tensor
parallelism (an all-reduce every layer), PP sends a single activation per stage
boundary, so it tolerates a Thunderbolt / Ethernet link between machines.

**Design.** The mlx_lm model files are untouched. vLLM's executor (Ray, or the
`mp` executor for a single node) owns the *control plane* — spawning one ranked
worker per stage. The *data plane* — the cross-stage activation hand-off — runs
over MLX's own `mx.distributed` backend (point-to-point `send` / `recv`),
not Ray. The default is TCP **ring**; [JACCL](#jaccl-rdma-pipeline-transport)
selects Thunderbolt RDMA explicitly. Rank 0 is the first stage (it embeds the tokens); rank `N-1` is the last
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

## JACCL RDMA pipeline transport

Select JACCL through `--additional-config`. vLLM sends this configuration to
every worker, including Ray workers on other Macs; shell variables on the
driver alone are not the transport configuration.

The included `tools/jaccl_two_rails.json` describes two Macs with these cables:

| Rank 0 device | Rank 1 device |
| --- | --- |
| `rdma_en1` | `rdma_en2` |
| `rdma_en2` | `rdma_en1` |

Edit the matrix to match your actual discovery results. Row `r`, column `p`
lists **rank r's local devices connected to rank p**, in matching lane order.
The diagonal is `null`; one lane can be a string, multiple lanes a list.
Do not sort interface names independently on each Mac.

Start a Ray cluster as above, with one `mlx` resource per Mac and a reachable
IPv4 `VLLM_HOST_IP` on each node. Ray needs connectivity in **both directions**;
an address that only works for a connection initiated by the other Mac is not
enough. These control addresses can be LAN addresses: the device matrix still
selects Thunderbolt RDMA for activations. Set `VLLM_HOST_IP` on the driver as
well as in each node's `ray start` environment. Then run on the driver:

```bash
RAY_ADDRESS=auto vllm serve Qwen/Qwen3-0.6B \
  --distributed-executor-backend ray \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 1 \
  --no-async-scheduling \
  --additional-config "$(cat tools/jaccl_two_rails.json)"
```

Requirements and behavior:

- RDMA must already be enabled, with a JACCL-capable MLX build. Each connected
  Thunderbolt interface must have its own IPv4 address / IPv4-mapped GID.
- Leave `GLOO_SOCKET_IFNAME` unset when starting both Ray nodes. On our Macs,
  forcing `en0` selected scoped link-local IPv6 addresses and stalled gloo peer
  setup. Multi-worker initialization now binds Gloo world and CPU subgroups
  to the explicit IPv4 `VLLM_HOST_IP`, avoiding macOS hostname changes when
  a bridge or network service is disabled.
- Export `MLX_METAL_FAST_SYNCH=1` **before starting both Ray nodes and the API
  driver**. MLX caches the fence selection; changing the variable requires new
  workers. The local UI launcher sets it on all three process launches. Setting
  it only on an API driver attached to existing Ray nodes is insufficient.
  This accelerates CPU/GPU fence synchronization; JACCL still uses CPU streams
  and receives finish on the host before downstream GPU work starts.
- Rank 0's discovered IPv4 address plus `coordinator_port` (default `59451`)
  forms the coordinator endpoint. Allow that TCP port between workers.
  Ray, gloo, and the coordinator use TCP for control; activations use RDMA.
- Initialization explicitly uses `backend="jaccl", strict=True`. Failure
  stops startup; it does **not** retry the TCP ring backend.
- JACCL receives finish on the CPU before the downstream GPU forward is
  submitted. This keeps a slow upstream stage from holding a Metal command
  buffer at a cross-stream fence long enough to hit the GPU watchdog. The
  activation buffer and transfer count stay the same; sends remain asynchronous.
- JACCL's *ring topology* enables its point-to-point/multi-rail implementation.
  This is separate from MLX's TCP backend named `ring`.
- Matrix shape, reciprocal lane counts, and a closed ring of 1–4 lanes per
  edge are checked before native initialization. More than two stages need
  the closing physical link from the last stage back to the first.
- Logs show pipeline rank-to-IP placement and each rank's device row. Match
  the matrix to that placement. Ray normally places rank 0 on the driver;
  remote rank order is not necessarily IP order and can change after restart.
- JACCL currently requires one pipeline stage per Mac and context parallel
  sizes of 1. Existing tensor-parallel and model restrictions still apply.
  GPT-OSS setup and validation are described below.
- Bootstrap uses temporary device files and restores environment variables
  afterwards. Restart worker processes to change transports or topology;
  MLX caches initialized groups within a process.

### macOS: Ray joins, then disappears with `No route to host`

A worker can register with the head while the head cannot make the reverse
connection needed for health checks. Check TCP from the **same environment that
starts Ray**, using a known listening port on the peer:

```bash
python -c 'import socket; socket.create_connection(("192.168.1.137", 22), timeout=3).close(); print("connected")'
nc -zv 192.168.1.137 22
```

Replace the address and port with your peer's. If `nc` succeeds while Python
immediately fails, investigate process permissions. In our reproduction,
Apple's Network framework reported `unsatisfiedReason=localNetworkDenied`.
After the user changed permissions, the same Python connection succeeded.

Open **System Settings → Privacy & Security → Local Network** and check the app
that launched the processes, such as an editor containing the terminal. macOS
attributes helper-process network access to the responsible app; see Apple's
[local network privacy documentation](https://developer.apple.com/documentation/technotes/tn3179-understanding-local-network-privacy).
Restart the test workers and verify both directions before retrying inference.

### Validate the transport without downloading a model

Run `tools/jaccl_pp_smoke.py` on both Macs with the same config and ordered
peer IPs. Rank 0 runs on the first IP, rank 1 on the second:

```bash
python tools/jaccl_pp_smoke.py --rank 0 \
  --peer-ips 10.0.0.1,10.0.0.2 --config tools/jaccl_two_rails.json
# On the second Mac, use the identical command with --rank 1.
```

This checks reductions, changed activation payloads in both directions, and
tiny Qwen3 pipeline logits against an unsplit reference, including cached
decode. Forward sends use `mx.async_eval`, matching the serving worker's
submission path. It is a correctness check; its output is not an inference
speed claim.

For GPT-OSS, use `--model gpt-oss` and set `VLLM_PP_LAYER_PARTITION=3,5` on
both commands. This uses an eight-layer tiny model with real routed experts,
alternating attention, and an eight-token sliding window. It checks chunked
prefill and cached decode across window boundaries, with 71 checks per rank.

### GPT-OSS pipeline parallelism

GPT-OSS uses alternating sliding-window and full-attention layers. Each stage
keeps the attention settings and KV caches for its own layers. Scheduler cache
names retain global layer numbers, so uneven splits cannot alias caches from
different stages. The model forward uses native GPT-OSS blocks and attention
sinks. Communication stays one activation handoff per stage boundary per
forward step; KV caches remain local.

The existing `mlx-community/gpt-oss-120b-MXFP4-Q8` checkpoint was validated on
48 GB and 64 GB Macs with 14 and 22 layers, respectively. After starting Ray as
above, use this driver command (replace the control IP and model path):

```bash
unset GLOO_SOCKET_IFNAME
RAY_ADDRESS=auto VLLM_HOST_IP=192.168.1.145 VLLM_PP_LAYER_PARTITION=14,22 \
  vllm serve "$HOME/.exo/models/mlx-community--gpt-oss-120b-MXFP4-Q8" \
    --served-model-name gpt-oss \
    --distributed-executor-backend ray \
    --pipeline-parallel-size 2 --tensor-parallel-size 1 \
    --no-async-scheduling --dtype bfloat16 \
    --reasoning-parser openai_gptoss \
    --gpu-memory-utilization 0.75 \
    --max-model-len 1024 --max-num-seqs 1 --max-num-batched-tokens 128 \
    --additional-config "$(cat tools/jaccl_two_rails.json)"
```

Both Macs need the same checkpoint and code. Set the same partition in both
Ray node environments too when launching workers independently. The context
and batching limits above are the initial correctness-test settings, not the
model's maximum. Keep TurboQuant disabled: quantized KV with attention sinks
is currently unsupported. Native MXFP4/Q8 **weight** quantization is supported.

### Validation recorded on 2026-09-17

Two M4 Max Macs (48 GB and 64 GB), macOS 27.0, MLX 0.32.1, MLX-LM
0.32.0, and vLLM 0.29.0 were tested with the included two-rail matrix:

- 65 checks passed on each rank: 20 reductions, 40 activation round trips
  (256 bytes and 4 MiB), and five tiny Qwen3 forward steps.
- Both synchronous and asynchronous forward submission passed. Prefill and
  four cached decode steps matched the unsplit model exactly (maximum absolute
  logit difference `0.0`).
- The non-slow test suite passed: 2,227 passed, five skipped, 24 deselected.
- Full Ray/API serving passed with Qwen3-0.6B, 14 layers per stage, and both
  workers reporting `backend=jaccl`. Ray 2.58.0 used LAN IPv4 control addresses;
  JACCL used the two Thunderbolt rails in the matrix.
- Three completion requests succeeded: a five-token prompt, a 1,031-token
  prompt with 512-token chunked prefill, and the repeated long prompt. Each
  generated 32 tokens; both long-prompt responses were identical at temperature
  zero. This verifies startup and generation, not sustained performance.
- GPT-OSS native-cache parity passed over both rails with a 3/5 split: all
  prefill/decode logits matched exactly as the sliding window advanced.
  Float32 and mixed MXFP4/Q8 unit tests also cover one-layer and odd splits.
- A six-layer BF16 GPT-OSS checkpoint served through vLLM with a 1/5 split.
  Both short and chunked-prefill requests produced exactly the same 12 greedy
  token IDs as native, unsplit MLX. This exercises the paged-cache path too.
- The existing GPT-OSS 120B MXFP4-Q8 checkpoint served three chat requests with
  a 14/22 split. Arithmetic returned `42`; a 439-token prompt crossing the
  128-token prefill chunk size returned the correct answer, and its repeat
  returned identical text. The test used a 1,024-token context limit.
- The initial Ray health-check failure was isolated to macOS local-network
  denial and cleared after a user permission change. A separate gloo startup
  stall was reproduced with `GLOO_SOCKET_IFNAME=en0` and resolved by leaving it
  unset. No Ray or gloo source changes were needed.

## Limitations

- **Checkpoint loading.** PP lazily slices MLX-LM safetensors; AWQ and GGUF are rejected because their loaders materialize the full model before slicing.
- **Co-located stages oversubscribe the KV budget.** Each stage applies `--gpu-memory-utilization` to the whole device independently — neither knows the other exists — so two stages on one Mac claim roughly twice the fraction. Lower it when stacking. Separate Macs are unaffected.
- **Synchronous scheduling required.** Run with `--no-async-scheduling` (the engine fails loud otherwise).
- **PP requires TP=1.** Combined PP+TP remains rejected. GPT-OSS TP2 is supported separately as described below.
- **Model support.** Uniform-attention models and GPT-OSS's alternating sliding/full attention are supported under PP. YOCO / hybrid / MLA / pooling / VLM / speculative decoding / LoRA remain rejected. GPT-OSS validation does not establish support for every sliding-window or MoE architecture.

## GPT-OSS tensor parallelism

The local two-Mac integration also supports GPT-OSS with `--tensor-parallel-size 2`
and `--pipeline-parallel-size 1`, using the Ray executor and synchronous scheduling.
Set `tensor_transport` in `--additional-config` to the same explicit JACCL options
and device matrix used by `pipeline_transport`. There is no TCP fallback.

```json
{"tensor_transport": {"backend": "jaccl", "coordinator_port": 59451,
 "device_matrix": [[null, ["rdma_en1", "rdma_en2"]],
                   [["rdma_en2", "rdma_en1"], null]]}}
```

Each worker lazily loads then applies MLX-LM's GPT-OSS `Model.shard`: every layer
remains present, with local attention/KV heads and expert projection dimensions.
The runner sizes paged KV caches for local heads. Both workers execute the same
scheduled batches; rank zero's chosen token IDs are synchronized over JACCL before
request state advances, including non-greedy sampling. vLLM reads rank zero's output.
Local-only decode-pipelining and selective-logit probes are disabled under TP.
Each tensor forward is evaluated before submitting the next prefill chunk, keeping
cross-rank collectives ordered and preventing overlapping GPU/CPU fence cycles.

Initial scope: two Macs, GPT-OSS MLX safetensors, DP=1, no combined PP, no LoRA,
no speculative decoding, no asynchronous scheduling or expert parallelism.
Unsupported configurations fail at admission. TP can require more memory on the
smaller Mac than an uneven pipeline split; size the cache budget accordingly.

The neighboring `inference-ui` project exposes a Pipeline-default selector with
an explicit Apply action. It reloads backend children while retaining the UI and
saved chats. Both modes have been exercised on the two-Mac GPT-OSS 120B checkpoint.

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

On a healthy boot each Mac logs `patched Ray V2 worker get_node_and_physical_gpu_ids ...` and an `EngineCore` actor is placed on each node IP.

**Design.** vLLM owns the whole DP control plane (replica placement, the DP coordinator, the request load balancer); `MetalPlatform` only relaxes admission to the supported shape. One Metal-specific detail: Ray honours a `worker_process_setup_hook` only from the **job** runtime_env (`ray.init`), and the DP engine manager connects to Ray without forwarding it — so vllm-metal registers the Apple-GPU worker patch at the job level itself before the engine connects (`MetalPlatform._register_dp_ray_worker_setup_hook`); otherwise the per-replica `RayWorkerProc` would `KeyError` on the custom `mlx` resource.

DP helps **under concurrency**, not single-stream latency, and stays below the 2× ideal because the head Mac also runs the API server, the DP coordinator, and the load balancer alongside its own replica; adding more replica nodes amortizes that head overhead. Measure your own throughput with `vllm bench serve`.

**Limitations.**

- **Capacity is unchanged.** Each replica loads the full model; DP does not serve a model larger than one Mac (that needs a sharded load, which is not yet available).
- **Dense models only.** MoE DP (expert-parallel all-to-all) is rejected — MLX has no `all_to_all` collective.
- **Requires a running Ray cluster.** DP registers the worker hook by initializing Ray itself; it fails loud if no cluster is reachable, or if Ray was already initialized by something else (do not pre-`ray.init` before serving).
- **Validated at 2 Macs.** More nodes (one replica each) should work but are untested.
- **Rejected combinations** (fail fast at config time): DP+PP, DP+TP, DP+MoE, DP+multimodal (the multimodal tensor-IPC path is DP=1 only), DP+speculative-decoding, DP+LoRA, DP+STT, `--data-parallel-external-lb` / `--data-parallel-hybrid-lb`, and any `--data-parallel-size-local` other than 1 (including the external-DP sentinel 0).
