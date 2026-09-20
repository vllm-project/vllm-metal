# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS expert parallelism: count-split experts, replicated routing."""

from __future__ import annotations

import logging
import os
from typing import cast

logger = logging.getLogger(__name__)


def expert_partition(world_size: int, num_experts: int) -> list[int]:
    """Per-rank expert counts from VLLM_METAL_EXPERT_PARTITION or an even split."""
    raw = os.getenv("VLLM_METAL_EXPERT_PARTITION")
    if raw is None:
        if num_experts % world_size:
            raise ValueError(
                f"{num_experts} experts do not divide evenly across {world_size} "
                "ranks; set VLLM_METAL_EXPERT_PARTITION."
            )
        return [num_experts // world_size] * world_size
    try:
        counts = [int(part) for part in raw.split(",")]
    except ValueError:
        counts = []
    if (
        len(counts) != world_size
        or sum(counts) != num_experts
        or any(count <= 0 for count in counts)
    ):
        raise ValueError(
            f"VLLM_METAL_EXPERT_PARTITION must be {world_size} positive integers "
            f"summing to {num_experts}; got {raw!r}."
        )
    return counts


def apply_expert_shard(model, tp) -> None:
    """Shard GPT-OSS for expert parallelism: attention as TP2, experts by count.

    Mirrors mlx_lm gpt_oss.Model.shard() for attention only — shard() itself
    width-slices experts, which expert parallelism must not do. Slicing runs
    before weight evaluation, on the same lazy lifecycle as apply_tensor_shard.
    Accepts either a TensorGroup or a bare mlx group.
    """
    import mlx.core as mx
    from mlx.nn.layers.distributed import shard_linear

    from vllm_metal import envs
    from vllm_metal.distributed.expert_decode import supports_experts

    group = getattr(tp, "group", tp)
    if getattr(model, "model_type", None) != "gpt_oss" or group.size() != 2:
        raise NotImplementedError("Expert sharding supports GPT-OSS on two Macs.")
    args = model.args
    if (
        args.num_attention_heads % group.size()
        or args.num_key_value_heads % group.size()
    ):
        raise ValueError("GPT-OSS attention heads must divide evenly across ranks.")
    counts = expert_partition(group.size(), args.num_local_experts)
    start = sum(counts[: group.rank()])
    end = start + counts[group.rank()]

    # Guard: ranks must agree on a contiguous partition covering every expert.
    # all_sum, not all_gather: this mlx/JACCL build delivers all_gather zeros
    # peer-to-peer on two rails (never exercised by the smoke or TP); all_sum
    # is the reduction the smoke proves in both directions. The peer's [start,
    # end] decodes as total minus mine; the squared terms catch a zeroed peer.
    # Skipped for non-distributed groups (single-process and unit-test fakes).
    if mx.distributed.is_available() and isinstance(group, mx.distributed.Group):
        mine = mx.array([start, end, start * start, end * end], dtype=mx.int32)
        total = mx.distributed.all_sum(mine, group=group, stream=mx.cpu)
        mx.eval(total)
        ts, te, ts_sq, te_sq = (int(v) for v in cast(list[int], total.tolist()))
        peer_start, peer_end = ts - start, te - end
        if (
            peer_start * peer_start != ts_sq - start * start
            or peer_end * peer_end != te_sq - end * end
            or peer_end <= peer_start
        ):
            raise ValueError(
                "Expert partitions disagree across ranks (check "
                "VLLM_METAL_EXPERT_PARTITION on every Mac); peer decoded as "
                f"[{peer_start}, {peer_end})."
            )
        if group.rank() == 0:
            if start != 0 or peer_start != end or peer_end != args.num_local_experts:
                raise ValueError(
                    "Expert partitions must be contiguous in rank order "
                    f"(rank 0 owns [{start}, {end}), peer [{peer_start}, "
                    f"{peer_end})); check VLLM_METAL_EXPERT_PARTITION."
                )
        else:
            if end != args.num_local_experts or peer_end != start or peer_start != 0:
                raise ValueError(
                    "Expert partitions must be contiguous in rank order "
                    f"(rank 1 owns [{start}, {end}), peer [{peer_start}, "
                    f"{peer_end})); check VLLM_METAL_EXPERT_PARTITION."
                )

    for layer in model.layers:
        attn = layer.self_attn
        attn.q_proj = shard_linear(attn.q_proj, sharding="all-to-sharded", group=group)
        attn.k_proj = shard_linear(attn.k_proj, sharding="all-to-sharded", group=group)
        attn.v_proj = shard_linear(attn.v_proj, sharding="all-to-sharded", group=group)
        attn.o_proj = shard_linear(attn.o_proj, sharding="sharded-to-all", group=group)
        attn.num_attention_heads //= group.size()
        attn.num_key_value_heads //= group.size()
        attn.num_key_value_groups = attn.num_attention_heads // attn.num_key_value_heads
        attn.sinks = attn.sinks[
            attn.num_attention_heads * group.rank() : attn.num_attention_heads
            * (group.rank() + 1)
        ]

        experts = layer.mlp.experts
        for proj in (experts.gate_proj, experts.up_proj, experts.down_proj):
            # Axis-zero slices can retain the full checkpoint allocation.
            # Materialize owned storage, as MLX's tensor sharder does.
            proj.weight = mx.contiguous(proj.weight[start:end])
            if "scales" in proj:
                proj.scales = mx.contiguous(proj.scales[start:end])
            if "bias" in proj:
                proj.bias = mx.contiguous(proj.bias[start:end])
            quant_biases = proj.get("biases")
            if quant_biases is not None:
                proj.biases = mx.contiguous(quant_biases[start:end])
        layer.mlp.sharding_group = group
        layer.mlp.expert_partition = (start, end)
        layer.mlp.expert_gpu_decode = (
            envs.VLLM_METAL_EP_GPU_DECODE and supports_experts(experts)
        )
    install_expert_routing()
    logger.info(
        "Expert GPU decode: rank=%d enabled_layers=%d/%d",
        group.rank(),
        sum(layer.mlp.expert_gpu_decode for layer in model.layers),
        len(model.layers),
    )


def install_expert_routing() -> None:
    """Route the replicated GPT-OSS router through rank-local experts.

    Both ranks see identical activations (post-o_proj all_sum), so the global
    top-k and softmax weights are identical; each rank zeroes the slots it
    does not own and points them at a dummy local expert, and the existing
    sharding_group all_sum sums the complementary partials — exact in exact
    arithmetic. Dummy slots keep shapes static; their zero weight cancels
    both expert output and bias.
    """
    import mlx.core as mx
    from mlx_lm.models import gpt_oss as mlx_gpt_oss

    from vllm_metal.distributed.expert_decode import decode_local_experts

    if getattr(mlx_gpt_oss.MLPBlock, "_ep_routing_installed", False):
        return
    original = mlx_gpt_oss.MLPBlock.__call__

    def ep_call(self, x):
        if getattr(self, "expert_partition", None) is None:
            return original(self, x)
        start, end = self.expert_partition
        g = self.router(x)
        scores, indices = mlx_gpt_oss.mlx_topk(g, k=self.num_experts_per_tok, axis=-1)
        expert_weights = mx.softmax(scores, axis=-1, precise=True)
        mask = (indices < start) | (indices >= end)
        local = mx.where(mask, mx.zeros_like(indices), indices - start)
        expert_weights = mx.where(mask, mx.zeros_like(expert_weights), expert_weights)
        if (
            getattr(self, "expert_gpu_decode", False)
            and indices.size == self.num_experts_per_tok
            and x.dtype in (mx.float16, mx.bfloat16)
        ):
            out = decode_local_experts(self.experts, x, indices, start, end)
        else:
            out = self.experts(x, local)
        out = out * mx.expand_dims(expert_weights, axis=-1)
        y = out.sum(axis=-2)
        return mx.distributed.all_sum(y, group=self.sharding_group)

    mlx_gpt_oss.MLPBlock.__call__ = ep_call
    mlx_gpt_oss.MLPBlock._ep_routing_installed = True
