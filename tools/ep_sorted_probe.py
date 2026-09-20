#!/usr/bin/env python3
"""One-shot hardware probe: tiny GPT-OSS, expert-parallel, big batch (sorted path)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx

from gpt_oss_smoke_model import tiny_model
from jaccl_pp_smoke import load_config
from vllm_metal.distributed.experts import apply_expert_shard
from vllm_metal.distributed.tensor import TensorGroup
from vllm_metal.distributed.transport import PipelineTransportConfig


def main() -> int:
    rank = int(sys.argv[1])
    peer_ips = sys.argv[2].split(",")
    config_path = Path(sys.argv[3])
    tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 128

    from argparse import Namespace

    ns = Namespace(config=config_path, peer_ips=",".join(peer_ips))
    _, transport = load_config(ns)
    group = transport.bootstrap_jaccl(rank, peer_ips)
    tg = TensorGroup(group)

    model = tiny_model()
    apply_expert_shard(model, tg)
    batch = mx.array(
        [[(7 * i + 3 * rank) % 90 + 1 for i in range(tokens)]], dtype=mx.int32
    )
    out = model(batch)
    mx.eval(out)
    print(
        f"rank {rank}: OK tokens={tokens} pairs={tokens * 2} "
        f"finite={bool(mx.all(mx.isfinite(out)).item())}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
