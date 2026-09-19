# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS expert parallelism: count-split experts, replicated routing."""

from __future__ import annotations

import os


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
    from mlx.nn.layers.distributed import shard_linear

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
            proj.weight = proj.weight[start:end]
            if "scales" in proj:
                proj.scales = proj.scales[start:end]
            if "bias" in proj:
                proj.bias = proj.bias[start:end]
            quant_biases = proj.get("biases")
            if quant_biases is not None:
                proj.biases = quant_biases[start:end]
        layer.mlp.sharding_group = group
        layer.mlp.expert_partition = (start, end)
