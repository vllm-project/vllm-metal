# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS tensor shards and strict JACCL communication across two Macs."""

from __future__ import annotations

from typing import Any, cast

from vllm_metal.distributed.transport import PipelineTransportConfig


def tensor_transport(config: Any) -> PipelineTransportConfig:
    options = (config.additional_config or {}).get("tensor_transport")
    if not isinstance(options, dict) or options.get("backend") != "jaccl":
        raise ValueError("Tensor parallelism requires explicit JACCL tensor_transport.")
    return PipelineTransportConfig.from_additional_config(
        {"pipeline_transport": options}, config.parallel_config.tensor_parallel_size
    )


def validate_tensor_config(config: Any) -> None:
    """Admit the tested TP2 topology; keep other distributed combinations closed."""
    parallel = config.parallel_config
    model = config.model_config
    unsupported = (
        parallel.tensor_parallel_size != 2
        or parallel.pipeline_parallel_size != 1
        or getattr(parallel, "data_parallel_size", 1) != 1
        or parallel.distributed_executor_backend != "ray"
        or model is None
        or getattr(getattr(model, "hf_config", None), "model_type", None) != "gpt_oss"
    )
    if unsupported:
        raise NotImplementedError(
            "Metal tensor parallelism currently requires GPT-OSS, TP=2, PP=1, "
            "DP=1, and the Ray executor; combined parallelism is not supported."
        )
    if (
        config.scheduler_config.async_scheduling
        or config.speculative_config is not None
        or config.lora_config is not None
        or getattr(model, "quantization", None) in ("gguf", "auto_awq")
        or getattr(model, "multimodal_config", None) is not None
        or getattr(model, "runner_type", "generate") != "generate"
    ):
        raise NotImplementedError(
            "Metal tensor parallelism requires synchronous GPT-OSS generation "
            "from MLX safetensors, without LoRA or speculative decoding."
        )
    tensor_transport(config)


class TensorGroup:
    """All ranks execute every layer; rank zero selects authoritative tokens."""

    def __init__(self, group: Any):
        self.group = group
        self.rank = group.rank()
        self.size = group.size()

    @classmethod
    def bootstrap(cls, rank: int, peer_ips: list[str], config: Any) -> TensorGroup:
        return cls(tensor_transport(config).bootstrap_jaccl(rank, peer_ips))

    def synchronize_tokens(self, token_ids: list[int]) -> list[int]:
        """Broadcast rank-zero samples before any rank advances request state.

        Nonzero ranks contribute zeros to a JACCL sum. This also handles random
        sampling: no assumption about matching per-process RNG consumption is made.
        """
        if not token_ids:
            return []
        import mlx.core as mx

        values = token_ids if self.rank == 0 else [0] * len(token_ids)
        with mx.stream(mx.cpu):
            result = mx.distributed.all_sum(
                mx.array(values, dtype=mx.int32), group=self.group, stream=mx.cpu
            )
            mx.eval(result)
        return cast(list[int], result.tolist())


def apply_tensor_shard(model: Any, tp: TensorGroup) -> None:
    """Shard lazy weights before evaluation, retaining all GPT-OSS layers."""
    if getattr(model, "model_type", None) != "gpt_oss" or tp.size != 2:
        raise NotImplementedError("Tensor sharding supports GPT-OSS on two Macs.")
    args = model.args
    if args.num_attention_heads % tp.size or args.num_key_value_heads % tp.size:
        raise ValueError("GPT-OSS attention heads must divide evenly across TP ranks.")
    model.shard(tp.group)
