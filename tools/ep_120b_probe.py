#!/usr/bin/env python3
"""120B EP probe: real checkpoint, EP shard, prefill + decode, no vLLM."""

import faulthandler

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from jaccl_pp_smoke import load_config
from vllm_metal.distributed.experts import apply_expert_shard
from vllm_metal.distributed.tensor import TensorGroup


def main() -> int:
    rank = int(sys.argv[1])
    peer_ips = sys.argv[2].split(",")
    model_path = Path(sys.argv[3])

    from argparse import Namespace

    backend = sys.argv[4] if len(sys.argv) > 4 else "jaccl"
    if backend == "ring":
        import os, tempfile
        from vllm_metal.distributed.pipeline import PipelineGroup

        with tempfile.NamedTemporaryFile("w", suffix=".hostfile", delete=False) as hf:
            import json as _json; hf.write(_json.dumps([[f"{ip}:{32323 + r}"] for r, ip in enumerate(peer_ips)]))
        os.environ["MLX_RANK"] = str(rank)
        os.environ["MLX_HOSTFILE"] = hf.name
        group = mx.distributed.init(backend="ring", strict=True)
        os.environ.pop("MLX_HOSTFILE", None)
        os.unlink(hf.name)
    else:
        import time as _time

        _, transport = load_config(Namespace(config=Path("tools/jaccl_two_rails.json"), peer_ips=",".join(peer_ips)))
        group = None
        for attempt in range(6):
            try:
                group = transport.bootstrap_jaccl(rank, peer_ips)
                break
            except RuntimeError:
                if attempt == 5:
                    raise
                print(f"rank {rank}: bootstrap retry {attempt + 1}", flush=True)
                _time.sleep(8)
    tg = TensorGroup(group)

    print(f"rank {rank}: loading {model_path}", flush=True)
    model, _ = load(str(model_path), lazy=True)
    mode = sys.argv[5] if len(sys.argv) > 5 else "ep"
    if mode == "tp":
        from vllm_metal.distributed.tensor import apply_tensor_shard

        apply_tensor_shard(model, tg)
        print(f"rank {rank}: TP control shard", flush=True)
    elif mode == "tpA":
        # TP weight geometry THROUGH the patched EP routing (full partition):
        # isolates patch code vs expert-count weight geometry.
        from vllm_metal.distributed.tensor import apply_tensor_shard
        from vllm_metal.distributed.experts import install_expert_routing

        apply_tensor_shard(model, tg)
        for layer in model.layers:
            layer.mlp.expert_partition = (0, model.args.num_local_experts)
        install_expert_routing()
        print(f"rank {rank}: TP geometry + EP routing patch", flush=True)
    else:
        apply_expert_shard(model, tg)
    print(f"rank {rank}: sharded (weights stay lazy, pulled per-op like serving)", flush=True)

    # Watchdog: dump the blocked Python stack every 120s without killing the
    # process, covering prefill AND decode. Identical repeated stacks at the
    # same line = deadlock; progressing lines = slow but moving.
    faulthandler.dump_traceback_later(120, repeat=True)
    import time as _time

    if len(sys.argv) > 6 and sys.argv[6] == "prefill-bench":
        # TTFT as context grows: fresh cache per length, prefill only.
        for length in (128, 512, 1024, 2048, 4096, 8192):
            cache = make_prompt_cache(model)
            batch = mx.array(
                [[1, 7, 11, 23] + [(3 * i) % 5000 + 10 for i in range(length - 4)]],
                dtype=mx.int32,
            )
            t0 = _time.perf_counter()
            out = model(batch, cache=cache)
            mx.eval(out)
            print(
                f"rank {rank}: TTFT ctx={length} in {_time.perf_counter()-t0:.2f}s",
                flush=True,
            )
        faulthandler.cancel_dump_traceback_later()
        print(f"rank {rank}: PREFILL BENCH DONE", flush=True)
        return 0

    cache = make_prompt_cache(model)
    batch = mx.array([[1, 7, 11, 23] + [(3 * i) % 5000 + 10 for i in range(124)]], dtype=mx.int32)
    t0 = _time.perf_counter()
    out = model(batch, cache=cache)
    mx.eval(out)
    print(f"rank {rank}: prefill OK in {_time.perf_counter()-t0:.1f}s finite={bool(mx.all(mx.isfinite(out)).item())}", flush=True)
    for step in range(4):
        t0 = _time.perf_counter()
        tok = mx.argmax(out[:, -1]).reshape(1, 1)
        out = model(mx.array([[int(tok.item())]], dtype=mx.int32), cache=cache)
        mx.eval(out)
        print(f"rank {rank}: decode {step} OK in {_time.perf_counter()-t0:.1f}s", flush=True)
    faulthandler.cancel_dump_traceback_later()
    print(f"rank {rank}: 120B EP PROBE PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
