# SPDX-License-Identifier: Apache-2.0
"""Checkpoint weight reads for vLLM's analytic memory traffic estimator."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_safetensors_params_metadata
from vllm.v1.metrics.perf import (
    AttentionMetrics,
    ExecutionContext,
    FfnMetrics,
    ParsedArgs,
    ParserChain,
    UnembedMetrics,
)

from vllm_metal import envs

logger = init_logger(__name__)
_FLOAT_BYTES = {"F16": 2, "BF16": 2, "F32": 4}


@lru_cache(maxsize=8)
def _checkpoint_metadata(
    model: str,
    revision: str | None,
    local_files: tuple[tuple[str, int, int], ...],
) -> dict[str, Any]:
    # local_files invalidates the cache when a local checkpoint is replaced.
    return get_safetensors_params_metadata(model, revision=revision)


def _get_metadata(config: VllmConfig) -> dict[str, Any]:
    model = config.model_config.model
    files = tuple(
        (str(path), stat.st_size, stat.st_mtime_ns)
        for path in sorted(Path(model).glob("*.safetensors"))
        if path.is_file()
        for stat in [path.stat()]
    )
    metadata = _checkpoint_metadata(model, config.model_config.revision, files)
    if not metadata:
        # A subsequent load may populate the offline cache.
        _checkpoint_metadata.cache_clear()
    return metadata


def _tensor_bytes(
    metadata: dict[str, Any], name: str, shape: tuple[int, int], *, packed: bool = False
) -> int | None:
    tensor = metadata.get(name)
    if tensor is None or tuple(tensor["shape"]) != shape:
        return None
    size = (
        {"U32": 4}.get(tensor["dtype"]) if packed else _FLOAT_BYTES.get(tensor["dtype"])
    )
    if size is None:
        return None
    start, end = tensor["data_offsets"]
    expected = shape[0] * shape[1] * size
    if end - start != expected:
        raise ValueError(f"Invalid safetensors byte count for {name}")
    return expected


def _projection_bytes(
    metadata: dict[str, Any],
    name: str,
    shape: tuple[int, int],
    quantization: dict[str, Any],
) -> int | None:
    # MLX-LM decides which modules to quantize from the presence of scales.
    if f"{name}.scales" not in metadata:
        return _tensor_bytes(metadata, f"{name}.weight", shape)

    quant = quantization.get(name, quantization)
    if not isinstance(quant, dict) or quant.get("mode", "affine") != "affine":
        return None
    bits, group_size = quant.get("bits"), quant.get("group_size")
    if (
        type(bits) is not int
        or bits <= 0
        or type(group_size) is not int
        or group_size <= 0
    ):
        raise ValueError("MLX quantization requires positive integer bits/group_size")
    rows, cols = shape
    if cols % group_size or cols * bits % 32:
        return None
    weight = _tensor_bytes(
        metadata, f"{name}.weight", (rows, cols * bits // 32), packed=True
    )
    scales = _tensor_bytes(metadata, f"{name}.scales", (rows, cols // group_size))
    biases = _tensor_bytes(metadata, f"{name}.biases", (rows, cols // group_size))
    if weight is None or scales is None or biases is None:
        return None
    return weight + scales + biases


class MLXCheckpointParser:
    """Map dense Llama/Qwen-style projections to checkpoint byte counts."""

    def __init__(self, component: str) -> None:
        self.component = component

    def _fallback(self, args: ParsedArgs, reason: str) -> ParsedArgs:
        logger.warning_once(
            "MLX MFU metrics cannot map %s checkpoint weights (%s); retaining "
            "upstream weight estimates for this component.",
            self.component,
            reason,
        )
        return args

    def parse(self, args: ParsedArgs, config: VllmConfig) -> ParsedArgs:
        model_config = config.model_config
        if envs.VLLM_METAL_BACKEND == "mps" or model_config.quantization is not None:
            return args
        quantization = getattr(model_config.hf_config, "quantization", None)
        if quantization is None:
            return args
        if not isinstance(quantization, dict):
            raise ValueError("MLX quantization must be a dictionary")

        parallel = config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            return self._fallback(args, "sharded execution")
        if quantization.get("mode", "affine") != "affine":
            return self._fallback(args, "non-affine quantization")

        specs: dict[str, list[tuple[str, tuple[int, int]]]] = {}
        hidden = args.hidden_size
        if self.component == "attn":
            q = args.num_attention_heads * args.head_dim
            kv = args.num_key_value_heads * args.head_dim
            for layer in range(args.num_hidden_layers):
                prefix = f"model.layers.{layer}.self_attn"
                specs.setdefault("qkv_weight", []).extend(
                    (f"{prefix}.{proj}_proj", (dim, hidden))
                    for proj, dim in (("q", q), ("k", kv), ("v", kv))
                )
                specs.setdefault("out_weight", []).append(
                    (f"{prefix}.o_proj", (hidden, q))
                )
        elif self.component == "ffn":
            if args.num_moe_layers:
                return self._fallback(args, "MoE routing")
            intermediate = args.intermediate_size
            for layer in range(args.num_hidden_layers):
                prefix = f"model.layers.{layer}.mlp"
                specs.setdefault("dense_up_gate_weights", []).extend(
                    (f"{prefix}.{proj}_proj", (intermediate, hidden))
                    for proj in ("up", "gate")
                )
                specs.setdefault("dense_down_weights", []).append(
                    (f"{prefix}.down_proj", (hidden, intermediate))
                )
        else:
            tied = getattr(model_config.hf_text_config, "tie_word_embeddings", False)
            head = "model.embed_tokens" if tied else "lm_head"
            specs["weight"] = [(head, (args.vocab_size, hidden))]

        metadata = _get_metadata(config)
        if not metadata:
            return self._fallback(args, "checkpoint metadata unavailable")
        reads = {}
        for operation, projections in specs.items():
            total = 0
            for name, shape in projections:
                size = _projection_bytes(metadata, name, shape, quantization)
                if size is None:
                    return self._fallback(args, f"unsupported tensor layout: {name}")
                total += size
            reads[operation] = total
        args.mlx_weight_reads = reads
        return args


class MLXAttentionMetrics(AttentionMetrics):
    mlx_weight_reads: dict[str, int] | None = None

    @classmethod
    def get_parser(cls) -> ParserChain:
        parser = super().get_parser()
        parser.add_parser(MLXCheckpointParser(cls.component_type()))
        return parser

    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        reads = super().get_read_bytes_breakdown(ctx, per_gpu)
        reads.update(self.mlx_weight_reads or {})
        return reads


class MLXFfnMetrics(FfnMetrics):
    mlx_weight_reads: dict[str, int] | None = None

    @classmethod
    def get_parser(cls) -> ParserChain:
        parser = super().get_parser()
        parser.add_parser(MLXCheckpointParser(cls.component_type()))
        return parser

    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        reads = super().get_read_bytes_breakdown(ctx, per_gpu)
        reads.update(self.mlx_weight_reads or {})
        return reads


class MLXUnembedMetrics(UnembedMetrics):
    mlx_weight_reads: dict[str, int] | None = None

    @classmethod
    def get_parser(cls) -> ParserChain:
        parser = super().get_parser()
        parser.add_parser(MLXCheckpointParser(cls.component_type()))
        return parser

    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        reads = super().get_read_bytes_breakdown(ctx, per_gpu)
        reads.update(self.mlx_weight_reads or {})
        return reads
